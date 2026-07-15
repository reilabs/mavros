//! Tools and utilities for computing layouts when generating bytecode.

use crate::{
    collections::HashMap,
    compiler::{
        codegen::constants,
        ssa::{
            ValueId,
            hlssa::{
                Constant, HLSSA, MAX_SUPPORTED_SIGNED_BITS, MAX_SUPPORTED_UNSIGNED_BITS, Type,
                TypeExpr,
            },
        },
        util::ice_non_elided_tuple,
    },
    vm::{self, bytecode},
};

// FRAME LAYOUTER
// ================================================================================================

/// Assists in computing the layout for a single virtual machine stack frame.
pub struct FrameLayouter {
    pub next_free: usize,
    pub variables: HashMap<ValueId, usize>,
}

impl FrameLayouter {
    pub fn new() -> Self {
        Self {
            next_free: 2,
            variables: HashMap::default(),
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
        assert!(size > 0 && size <= MAX_SUPPORTED_UNSIGNED_BITS);
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += int_cell_count(size);
        bytecode::FramePosition(r)
    }

    pub fn alloc_field(&mut self, value: ValueId) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += bytecode::FELT_LIMBS;
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
        self.next_free += bytecode::FELT_LIMBS;
        bytecode::FramePosition(r)
    }

    pub fn alloc_ptr(&mut self, value: ValueId) -> bytecode::FramePosition {
        self.variables.insert(value, self.next_free);
        let r = self.next_free;
        self.next_free += 1;
        bytecode::FramePosition(r)
    }

    /// Returns the size of `tp` in u`64` cells.
    pub fn type_size(&self, tp: &Type) -> usize {
        match tp.expr {
            TypeExpr::Field => bytecode::FELT_LIMBS,
            TypeExpr::U(bits) => {
                assert!(bits <= MAX_SUPPORTED_UNSIGNED_BITS);
                int_cell_count(bits)
            }
            TypeExpr::I(bits) => {
                assert!(
                    bits <= MAX_SUPPORTED_SIGNED_BITS,
                    "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
                );
                1
            }
            TypeExpr::Array(_, _) => constants::POINTER_SIZE_CELLS,
            TypeExpr::Slice(_) => constants::POINTER_SIZE_CELLS,
            TypeExpr::WitnessOf(_) => constants::POINTER_SIZE_CELLS,
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
            TypeExpr::Ref(_) => constants::POINTER_SIZE_CELLS,
            // Blobs are stored by value, inline in the frame.
            TypeExpr::Blob(ref elem, n) => n * self.type_size(elem),
            _ => todo!(),
        }
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

// STRUCT LAYOUT INTERNER
// ================================================================================================

/// Interns unique struct shapes during codegen and returns a stable index into the resulting
/// descriptor table.
pub struct StructLayoutInterner {
    pub table: Vec<vm::array::StructDescriptor>,
    pub index: HashMap<Vec<(u32, bool)>, usize>,
}

impl StructLayoutInterner {
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            index: HashMap::default(),
        }
    }

    pub fn intern(&mut self, fields: Vec<(u32, bool)>) -> usize {
        if let Some(&idx) = self.index.get(&fields) {
            return idx;
        }
        let idx = self.table.len();
        self.table
            .push(vm::array::StructDescriptor::new(fields.clone()));
        self.index.insert(fields, idx);
        idx
    }

    pub fn into_table(self) -> Vec<vm::array::StructDescriptor> {
        self.table
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

/// Visit the `u64` words of `constant` in frame order — the order in which both the constant pool
/// and a `MovConst` spill lay them out.
///
/// This is the single source of truth for constant word ordering: [`ConstantPoolInterner`], the
/// `spill_constant_to_frame` inline path, and `constant_cell_count` (which just counts the visits)
/// all drive it, so the pooled and inline representations — and the memcpy size — cannot drift out
/// of sync. Adding a new [`Constant`] variant only requires updating this function.
pub fn for_each_constant_word(constant: &Constant, visit: &mut impl FnMut(u64)) {
    match constant {
        Constant::U(size, val) => match size {
            bits if *bits <= 64 => visit(*val as u64),
            128 => {
                visit(*val as u64);
                visit((*val >> 64) as u64);
            }
            bits => panic!("unsupported unsigned integer width: {bits}"),
        },
        Constant::I(size, val) => {
            assert!(
                *size <= MAX_SUPPORTED_SIGNED_BITS,
                "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
            );
            visit(*val as u64);
        }
        Constant::Field(val) => {
            for i in 0..bytecode::FELT_LIMBS {
                visit(val.0.0[i]);
            }
        }
        Constant::Blob(blob) => {
            for element in &blob.elements {
                for_each_constant_word(element, visit);
            }
        }
        Constant::FnPtr(_) => panic!("FnPtr constants not supported in codegen"),
    }
}

// GLOBAL FRAME LAYOUTER
// ================================================================================================

pub struct GlobalFrameLayouter {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
    pub total_size: usize,
}

impl GlobalFrameLayouter {
    pub fn new(ssa: &HLSSA) -> Self {
        let global_types = ssa.get_global_types();
        let mut offsets = Vec::new();
        let mut sizes = Vec::new();
        let mut next_free = 0usize;
        for typ in global_types.iter() {
            let size = Self::type_frame_size(typ);
            offsets.push(next_free);
            sizes.push(size);
            next_free += size;
        }
        GlobalFrameLayouter {
            offsets,
            sizes,
            total_size: next_free,
        }
    }

    fn type_frame_size(typ: &Type) -> usize {
        match &typ.expr {
            TypeExpr::Field => bytecode::FELT_LIMBS,
            TypeExpr::U(bits) => {
                assert!(*bits <= MAX_SUPPORTED_UNSIGNED_BITS);
                int_cell_count(*bits)
            }
            TypeExpr::I(bits) => {
                assert!(
                    *bits <= MAX_SUPPORTED_SIGNED_BITS,
                    "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
                );
                1
            }
            // Heap-allocated types are pointers (1 word)
            _ => 1,
        }
    }

    pub fn get_offset(&self, global: usize) -> usize {
        self.offsets[global]
    }

    pub fn get_size(&self, global: usize) -> usize {
        self.sizes[global]
    }
}

pub fn int_cell_count(bits: usize) -> usize {
    assert!(bits > 0 && bits <= MAX_SUPPORTED_UNSIGNED_BITS);
    bits.div_ceil(64)
}
