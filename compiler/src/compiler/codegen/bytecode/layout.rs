//! Tools and utilities for computing layouts when generating bytecode.

use super::WitnessMode;
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{IntBits, int_bits::HOST_LIMB_BITS};

use crate::{
    collections::HashMap,
    compiler::{
        codegen::constants,
        ssa::{
            ValueId,
            hlssa::{Constant, HLSSA, MAX_SUPPORTED_INT_BITS, Type, TypeExpr},
        },
        util::ice_non_elided_tuple,
    },
    vm::bytecode,
};

// FRAME LAYOUTER
// ================================================================================================

/// Assists in computing the layout for a single virtual machine stack frame.
pub struct FrameLayouter {
    pub next_free: usize,
    mode: WitnessMode,
    pub variables: HashMap<ValueId, usize>,

    /// The cells one element of the program's field occupies.
    field_cells: usize,
}

impl FrameLayouter {
    pub fn new(field: FieldConfig, mode: WitnessMode) -> Self {
        Self::with_field_cells(field.storage_cells(), mode)
    }

    pub(super) fn with_field_cells(field_cells: usize, mode: WitnessMode) -> Self {
        Self {
            next_free: 2,
            mode,
            variables: HashMap::default(),
            field_cells,
        }
    }

    pub fn get_value(&mut self, value: ValueId) -> bytecode::FramePosition {
        bytecode::FramePosition(self.variables[&value])
    }

    pub fn alloc_value(&mut self, value: ValueId, tp: &Type) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += self.type_size(&tp);
        bytecode::FramePosition(r)
    }

    pub fn alloc_int(&mut self, value: ValueId, size: usize) -> bytecode::FramePosition {
        assert!(size > 0 && size <= MAX_SUPPORTED_INT_BITS);
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += int_cell_count(size);
        bytecode::FramePosition(r)
    }

    pub fn alloc_field(&mut self, value: ValueId) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += self.field_cells;
        bytecode::FramePosition(r)
    }

    pub fn alloc_long_data(&mut self, value: ValueId, cells: usize) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += cells;
        bytecode::FramePosition(r)
    }

    pub fn alloc_temp_field(&mut self) -> bytecode::FramePosition {
        let r = self.next_free;
        self.next_free += self.field_cells;
        bytecode::FramePosition(r)
    }

    pub fn alloc_ptr(&mut self, value: ValueId) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += 1;
        bytecode::FramePosition(r)
    }

    /// Returns the size of `tp` in `u64` cells.
    pub fn type_size(&self, tp: &Type) -> usize {
        slot_cells(tp, self.field_cells, self.mode)
    }

    pub fn alloc_scratch(&mut self, count: usize) -> bytecode::FramePosition {
        let r = self.next_free;
        self.next_free += count;
        bytecode::FramePosition(r)
    }

    // This method needs to ensure contiguous storage!
    pub fn alloc_many_contiguous(
        &mut self,
        values: Vec<(ValueId, &Type)>,
    ) -> bytecode::FramePosition {
        let r = self.next_free;
        for (value, tp) in values {
            self.alloc_value(value, tp);
        }
        bytecode::FramePosition(r)
    }
}

// CONSTANT POOL INTERNER
// ================================================================================================

/// Interns multi-cell constants during codegen into a program-global pool and returns a stable
/// word offset into it.
///
/// The pool stores each distinct constant's words once, in the exact frame layout that
/// `spill_constant_to_frame` would have written, so a `mov_const_pool` memcpy into a frame slot is
/// byte-identical to the equivalent `MovConst` chain. Constants are keyed by their interned
/// `ValueId` (bijective with the constant's value), which is cheaper than hashing the value and
/// gives program-global deduplication for free.
pub struct ConstantPoolInterner {
    pub pool: Vec<u64>,
    pub index: HashMap<ValueId, usize>,
}

impl ConstantPoolInterner {
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            index: HashMap::default(),
        }
    }

    /// Intern `constant` (identified by `vid`) and return its word offset in the pool. On a miss
    /// the constant's words are appended in frame order.
    pub fn intern(&mut self, vid: ValueId, constant: &Constant) -> usize {
        if let Some(&off) = self.index.get(&vid) {
            return off;
        }
        let off = self.pool.len();
        let pool = &mut self.pool;
        for_each_constant_word(constant, &mut |word| pool.push(word));
        self.index.insert(vid, off);
        off
    }

    pub fn into_pool(self) -> Vec<u64> {
        self.pool
    }
}

/// Visit the `u64` words of `constant` in frame order.
///
/// The definition of constant word order, used by [`ConstantPoolInterner`],
/// `spill_constant_to_frame`'s inline path and `constant_cell_count`, so the pooled and inline
/// forms and the memcpy size cannot drift apart. A new [`Constant`] variant must be handled here.
pub fn for_each_constant_word(constant: &Constant, visit: &mut impl FnMut(u64)) {
    match constant {
        Constant::Int(v) => {
            // A normalized pattern _is_ the word sequence, at the frame's length. `int_cell_count`
            // is called anyway for the width cap it asserts.
            assert_eq!(int_cell_count(v.bits()), v.limb_count());
            for &word in v.limbs() {
                visit(word);
            }
        }
        Constant::Field(val) => {
            // FIELD-ASSUMPTION: L3-frame
            for limb in val.montgomery_limbs() {
                visit(limb);
            }
        }
        Constant::Blob(blob) => {
            for element in &blob.elements {
                for_each_constant_word(element, visit);
            }
        }
        Constant::FnPtr(_) => ice!("FnPtr constants not supported in codegen"),
    }
}

// GLOBAL FRAME LAYOUTER
// ================================================================================================

pub struct GlobalFrameLayouter {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
    pub total_size: usize,
    witnesses: Vec<bool>,
}

impl GlobalFrameLayouter {
    pub fn new(ssa: &HLSSA, mode: WitnessMode) -> Self {
        let global_types = ssa.get_global_types();
        let field_cells = ssa.field().storage_cells();
        let mut offsets = Vec::new();
        let mut sizes = Vec::new();
        let mut next_free = 0usize;
        for typ in global_types.iter() {
            let size = slot_cells(typ, field_cells, mode);
            offsets.push(next_free);
            sizes.push(size);
            next_free += size;
        }
        GlobalFrameLayouter {
            offsets,
            sizes,
            total_size: next_free,
            witnesses: global_types.iter().map(|ty| ty.is_witness_of()).collect(),
        }
    }

    pub fn is_witness(&self, global: usize) -> bool {
        self.witnesses[global]
    }

    pub fn get_offset(&self, global: usize) -> usize {
        self.offsets[global]
    }

    pub fn get_size(&self, global: usize) -> usize {
        self.sizes[global]
    }
}

/// The `u64` cells a value of type `tp` occupies in a frame or in the globals, where a field
/// element takes `field_cells` and witnesses use the representation selected by `mode`.
fn slot_cells(tp: &Type, field_cells: usize, mode: WitnessMode) -> usize {
    match &tp.expr {
        TypeExpr::Field => field_cells,
        TypeExpr::WitnessOf(_) if mode == WitnessMode::Verifier => field_cells,
        // Width-driven, and it must stay that way: `alloc_int` advances the frame by
        // `int_cell_count`, so a second formula here would silently disagree with it for any width
        // needing more than one cell.
        TypeExpr::Int(bits) => {
            assert!(*bits <= MAX_SUPPORTED_INT_BITS);
            int_cell_count(*bits)
        }
        TypeExpr::Array(_, _) | TypeExpr::Slice(_) | TypeExpr::WitnessOf(_) | TypeExpr::Ref(_) => {
            constants::POINTER_SIZE_CELLS
        }
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        // Blobs are stored by value, inline in the frame.
        TypeExpr::Blob(elem, n) => n * slot_cells(elem, field_cells, mode),
        TypeExpr::Function(_) => ice!("function types are erased before codegen: {tp}"),
    }
}

/// The number of VM frame cells a `bits`-wide integer occupies.
pub fn int_cell_count(bits: usize) -> usize {
    assert!(bits <= MAX_SUPPORTED_INT_BITS);
    IntBits::limbs_for_bits(bits)
}

/// The widest container the `SpreadU32ToU64` opcode interleaves, and half the widest
/// `UnspreadU64ToU32` separates; a wider one takes the `_intn` pair.
pub const SPREAD_MAX_BITS: usize = 32;

/// The widest integer the cell lane holds: one frame cell, which is one host `u64`.
const CELL_LANE_BITS: usize = HOST_LIMB_BITS;

/// The widest integer the double lane holds: two cells, which is one host `u128`.
pub const DOUBLE_LANE_BITS: usize = 2 * HOST_LIMB_BITS;

/// Which of the interpreter's three integer lanes a width is lowered to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lane {
    /// One cell, holding `1..=64` bits, computed using a host `u64` for efficiency.
    Cell,

    /// Two cells, holding `65..=128` bits, computed using a host `u128` for efficiency.
    Double,

    /// [`int_cell_count`] cells, computed limb by limb.
    Wide,
}

/// The lane a `bits`-wide integer is lowered to.
///
/// # Panics
///
/// If `bits` is zero or above [`MAX_SUPPORTED_INT_BITS`]. [`FrameLayouter::alloc_int`] refuses both
/// where a value of one is allocated.
pub fn int_lane(bits: usize) -> Lane {
    assert!(
        bits <= MAX_SUPPORTED_INT_BITS,
        "ICE: an int{bits} is wider than the {MAX_SUPPORTED_INT_BITS}-bit maximum"
    );
    match bits {
        0 => ice!("Encountered a query for a zero-width integer lane"),
        1..=CELL_LANE_BITS => Lane::Cell,
        ..=DOUBLE_LANE_BITS => Lane::Double,
        _ => Lane::Wide,
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_and_global_witness_blobs_have_the_same_layout() {
        let mut ssa = HLSSA::new();
        let witness = Type::witness_of(Type::field());
        ssa.set_global_types(vec![
            witness.clone(),
            Type::blob(witness.clone(), 3),
            Type::blob(Type::blob(witness, 2), 3),
            Type::field(),
        ]);
        for mode in [WitnessMode::Ad, WitnessMode::Verifier] {
            let local = FrameLayouter::new(ssa.field(), mode);
            let global = GlobalFrameLayouter::new(&ssa, mode);
            let witness_cells = match mode {
                WitnessMode::Ad => constants::POINTER_SIZE_CELLS,
                WitnessMode::Verifier => ssa.field().storage_cells(),
            };
            let expected = [
                witness_cells,
                3 * witness_cells,
                6 * witness_cells,
                ssa.field().storage_cells(),
            ];
            let mut offset = 0;
            for (i, (ty, size)) in ssa.get_global_types().iter().zip(expected).enumerate() {
                assert_eq!(local.type_size(ty), size);
                assert_eq!(global.get_size(i), size);
                assert_eq!(global.get_offset(i), offset);
                offset += size;
            }
            assert_eq!(global.total_size, offset);
        }
    }

    #[test]
    fn a_width_past_the_old_cap_is_laid_out_rather_than_refused() {
        // No corpus program reaches these widths, so these asserts pin the layout there:
        // `IntBits::limbs_for_bits` is width-generic, and what is checked is that the bound in
        // front of it admits the width.
        assert_eq!(int_cell_count(200), 4, "200 bits is four 64-bit cells");
        assert_eq!(int_cell_count(MAX_SUPPORTED_INT_BITS), 256);
        assert_eq!(
            FrameLayouter::new(FieldConfig::bn254(), WitnessMode::Ad)
                .type_size(&Type::int(MAX_SUPPORTED_INT_BITS)),
            256
        );
    }

    #[test]
    fn the_two_frame_size_formulas_agree_at_every_width() {
        // `alloc_value` uses `type_size` and `alloc_int` uses `int_cell_count`, and a value can be
        // allocated by either, so the two must agree at every width.
        let layouter = FrameLayouter::new(FieldConfig::bn254(), WitnessMode::Ad);
        for bits in [
            1usize,
            8,
            32,
            63,
            64,
            65,
            127,
            128,
            129,
            192,
            1000,
            MAX_SUPPORTED_INT_BITS - 1,
            MAX_SUPPORTED_INT_BITS,
        ] {
            assert_eq!(
                layouter.type_size(&Type::int(bits)),
                int_cell_count(bits),
                "type_size and alloc_int disagree at {bits} bits"
            );
        }
    }

    #[test]
    fn a_field_slot_is_the_fields_storage_cells() {
        let bn254 = FieldConfig::bn254();
        assert_eq!(
            FrameLayouter::new(bn254, WitnessMode::Ad).type_size(&Type::field()),
            bn254.storage_cells()
        );

        // Three cells, which no field has, so a slot sized by a literal fails.
        let mut layouter = FrameLayouter::with_field_cells(3, WitnessMode::Ad);
        assert_eq!(layouter.type_size(&Type::field()), 3);
        assert_eq!(layouter.type_size(&Type::blob(Type::field(), 2)), 6);
        let named = layouter.alloc_field(ValueId(0));
        let temp = layouter.alloc_temp_field();
        let after = layouter.alloc_temp_field();
        assert_eq!([named.0, temp.0, after.0], [2, 5, 8]);
    }

    #[test]
    fn a_constant_occupies_as_many_words_as_its_width_needs() {
        // The word count follows the width, and `alloc_int` must agree: where they differ, a wide
        // constant loses its upper word.
        let count = |c: &Constant| {
            let mut n = 0;
            for_each_constant_word(c, &mut |_| n += 1);
            n
        };
        for bits in [1usize, 8, 32, 63, 64] {
            assert_eq!(count(&Constant::int(bits, 1)), 1, "at {bits} bits");
        }
        assert_eq!(count(&Constant::int(128, 1)), 2);

        for bits in [65usize, 96, 127] {
            assert_eq!(count(&Constant::int(bits, 1)), 2, "at {bits} bits");
        }

        // The upper word is emitted, not dropped: a payload living entirely above bit 64 must
        // still produce two words, the low one zero.
        let mut words = Vec::new();
        for_each_constant_word(&Constant::int(128, 1u128 << 100), &mut |w| words.push(w));
        assert_eq!(words, vec![0u64, 1u64 << 36]);

        // The count matches what the frame reserves for the width, wide widths included, where a
        // host-word-shaped emitter would write two words into a four-cell slot.
        for bits in [1usize, 8, 32, 63, 64, 65, 96, 127, 128, 200, 1000] {
            assert_eq!(
                count(&Constant::int(bits, 1)),
                int_cell_count(bits),
                "constant words and frame cells disagree at {bits} bits"
            );
        }
    }
}
