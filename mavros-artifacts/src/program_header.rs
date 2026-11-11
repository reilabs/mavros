//! The header every VM program binary opens with. The compiler writes it, the VM reads it, and an
//! [`Artifact`] checks it against its R1CS and manifest when it is built or read. The opcode stream
//! after it is not validated.
//!
//! [`Artifact`]: crate::Artifact

use crate::{ConstraintsLayout, FieldId, WitnessLayout};

/// Index of the witness-generation entry point in a program's entry table.
pub const ENTRY_WITGEN: usize = 0;
/// Index of the AD entry point in a program's entry table.
pub const ENTRY_AD: usize = 1;
/// Index of the direct forward verifier entry point.
pub const ENTRY_VERIFIER: usize = 2;
/// An absent entry in a program's entry table.
pub const NO_ENTRY_POINT: usize = usize::MAX;
/// The word [`NO_ENTRY_POINT`] is serialized as, the same on every target.
pub const NO_ENTRY_POINT_WORD: u64 = u64::MAX;
/// The word that opens every function, followed by its frame size and then its opcodes.
pub const FUNCTION_MARKER: u64 = u64::MAX;

/// The first word of every program binary: `MAVROSVM` as big-endian ASCII.
pub const PROGRAM_MAGIC: u64 = u64::from_be_bytes(*b"MAVROSVM");

/// The program binary's format version, its second word. Changing existing production opcode
/// numbers or operands, the `BoxedLayout` encoding, the function marker, or the header layout must
/// bump this.
/// Appending opcodes without changing existing production encodings does not require a bump.
/// The third word names the program's field ([`FieldId::code`]).
pub const PROGRAM_FORMAT_VERSION: u64 = 5;

/// Buffer layouts bound to the compiled constraint traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintLayouts([u64; 9]);

impl ConstraintLayouts {
    pub fn new(witness: WitnessLayout, constraints: ConstraintsLayout) -> Self {
        Self([
            witness.algebraic_size as u64,
            witness.multiplicities_size as u64,
            witness.challenges_size as u64,
            witness.logup_repetitions as u64,
            witness.tables_data_size as u64,
            witness.lookups_data_size as u64,
            constraints.algebraic_size as u64,
            constraints.tables_data_size as u64,
            constraints.lookups_data_size as u64,
        ])
    }

    /// The nine header words, in the order [`Self::new`] reads its fields.
    pub fn words(self) -> [u64; 9] {
        self.0
    }
}

/// The decoded header of a program binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramHeader {
    /// The field the program computes over.
    pub field: FieldId,
    /// Layouts emitted by the compiler, checked before constraint execution.
    pub constraint_layouts: Option<ConstraintLayouts>,
    /// Interned constant pool (flat words), read at runtime by `mov_const_pool`.
    pub constant_pool: Vec<u64>,
    pub global_frame_size: usize,
    /// The exact number of input fields the caller must supply.
    pub entry_blob_field_count: usize,
    /// Absolute word offsets of each entry point's function marker, in entry-table order
    /// ([`ENTRY_WITGEN`], [`ENTRY_AD`], ...). The entry's frame size lives at `offset + 1` and
    /// its first opcode at `offset + 2`. [`NO_ENTRY_POINT`] marks an absent entry.
    pub entry_points: Vec<usize>,
    /// Word offset of the first function marker, i.e. where the opcode stream begins.
    pub code_start: usize,
}

impl ProgramHeader {
    /// Whether the program was compiled for exactly these layouts.
    pub fn has_layouts(&self, witness: WitnessLayout, constraints: ConstraintsLayout) -> bool {
        self.constraint_layouts == Some(ConstraintLayouts::new(witness, constraints))
    }

    /// Decodes a program binary's field, layouts, constant pool, global frame size, entry blob
    /// field count and entry-point table.
    ///
    /// # Errors
    ///
    /// If the binary does not open with [`PROGRAM_MAGIC`] followed by [`PROGRAM_FORMAT_VERSION`]:
    /// a binary of any other format numbers its opcodes differently, so they would select the
    /// wrong handlers. If its field word names no supported field, it ends inside the header, or
    /// an entry point is not on a function marker with room for a frame size and an opcode.
    pub fn parse(program: &[u64]) -> Result<Self, String> {
        if program.first() != Some(&PROGRAM_MAGIC) {
            return Err("not a mavros VM program binary".to_string());
        }
        let version = program.get(1).copied();
        if version != Some(PROGRAM_FORMAT_VERSION) {
            return Err(format!(
                "unsupported VM program format version {version:?}; expected {PROGRAM_FORMAT_VERSION}"
            ));
        }
        // A length too wide for `usize` runs past the end of any program this target can hold.
        let words = |start: usize, len: u64| {
            usize::try_from(len)
                .ok()
                .and_then(|len| start.checked_add(len))
                .and_then(|end| program.get(start..end))
                .ok_or_else(|| "the VM program ends inside its header".to_string())
        };
        let word = |index: usize| words(index, 1).map(|words| words[0]);

        let field_word = word(2)?;
        let field = FieldId::from_code(field_word).ok_or_else(|| {
            format!("the VM program names no supported field (word {field_word})")
        })?;
        let (constraint_layouts, off) = match word(3)? {
            0 => (None, 4),
            1 => {
                let layouts = words(4, 9)?.try_into().expect("9 words");
                (Some(ConstraintLayouts(layouts)), 13)
            }
            tag => return Err(format!("invalid constraint layout tag {tag}")),
        };
        let constant_pool = words(off + 1, word(off)?)?.to_vec();
        let off = off + 1 + constant_pool.len();
        let global_frame_size = to_usize(word(off)?, "global frame size")?;
        let entry_blob_field_count = to_usize(word(off + 1)?, "entry blob field count")?;
        let entry_words = words(off + 3, word(off + 2)?)?;
        let code_start = off + 3 + entry_words.len();
        let mut entry_points = Vec::with_capacity(entry_words.len());
        for (index, &entry_word) in entry_words.iter().enumerate() {
            if entry_word == NO_ENTRY_POINT_WORD {
                entry_points.push(NO_ENTRY_POINT);
                continue;
            }
            // The VM reads the frame size at `entry + 1` and starts at `entry + 2`, unchecked. A
            // word too wide for `usize` is past the end of any program this target can hold.
            let entry = usize::try_from(entry_word)
                .ok()
                .filter(|entry| (code_start..program.len() - 2).contains(entry))
                .ok_or_else(|| {
                    format!(
                        "the VM program's entry point {index} at word {entry_word} lies outside \
                         its code"
                    )
                })?;
            if program[entry] != FUNCTION_MARKER {
                return Err(format!(
                    "the VM program's entry point {index} at word {entry} is not a function marker"
                ));
            }
            entry_points.push(entry);
        }
        Ok(ProgramHeader {
            field,
            constraint_layouts,
            constant_pool,
            global_frame_size,
            entry_blob_field_count,
            entry_points,
            code_start,
        })
    }
}

/// A header word as a size, refused when it is too wide for this target's `usize`.
fn to_usize(word: u64, what: &str) -> Result<usize, String> {
    usize::try_from(word)
        .map_err(|_| format!("the VM program's {what} {word} does not fit this target's usize"))
}
