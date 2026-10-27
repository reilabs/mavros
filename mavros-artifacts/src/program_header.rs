//! The header every VM program binary opens with.
//!
//! The compiler writes it, the VM reads it before dispatch, and [`Artifact::from_bytes`] checks
//! the bytecode an artifact carries against its R1CS and manifest through it. It lives here,
//! below both, because it is their shared contract and encodes this crate's [`WitnessLayout`] and
//! [`ConstraintsLayout`].
//!
//! [`Artifact::from_bytes`]: crate::Artifact::from_bytes

use crate::{ConstraintsLayout, FieldId, WitnessLayout};

/// Index of the witness-generation entry point in a program's entry table.
pub const ENTRY_WITGEN: usize = 0;
/// Index of the AD entry point in a program's entry table.
pub const ENTRY_AD: usize = 1;
/// Index of the direct forward verifier entry point.
pub const ENTRY_VERIFIER: usize = 2;
/// An absent entry in a program or serialized entry table.
pub const NO_ENTRY_POINT: usize = usize::MAX;

/// The first word of every program binary: `MAVROSVM` as big-endian ASCII.
pub const PROGRAM_MAGIC: u64 = u64::from_be_bytes(*b"MAVROSVM");

/// The program binary's format version, its second word. Changing existing production opcode
/// numbers or operands, the `BoxedLayout` encoding, or the header layout must bump this.
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
    /// wrong handlers. If its field word names no supported field, or it ends inside the header.
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
        let words = |start: usize, len: usize| {
            start
                .checked_add(len)
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
        let pool_len = word(off)? as usize;
        let constant_pool = words(off + 1, pool_len)?.to_vec();
        let off = off + 1 + pool_len;
        let global_frame_size = word(off)? as usize;
        let entry_blob_field_count = word(off + 1)? as usize;
        let num_entries = word(off + 2)? as usize;
        let entry_points = words(off + 3, num_entries)?
            .iter()
            .map(|&offset| offset as usize)
            .collect();
        Ok(ProgramHeader {
            field,
            constraint_layouts,
            constant_pool,
            global_frame_size,
            entry_blob_field_count,
            entry_points,
            code_start: off + 3 + num_entries,
        })
    }
}
