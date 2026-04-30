use crate::compiler::ir::r#type::SSAType;
use crate::compiler::ssa::{Block, Function, FunctionId, Instruction, SSA, ValueId};
use itertools::Itertools;
use std::fmt::{self, Display, Formatter};

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/// SSA value type for LLSSA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LLType {
    /// Sized unsigned integer. Int(1) = bool, Int(8) = byte, Int(32), Int(64).
    Int(u32),
    /// Opaque pointer.
    Ptr,
    /// Multi-word aggregate, by value. Only for value-safe structs.
    Struct(LLStruct),
}

impl LLType {
    pub fn i1() -> Self {
        LLType::Int(1)
    }
    pub fn i32() -> Self {
        LLType::Int(32)
    }
    pub fn i64() -> Self {
        LLType::Int(64)
    }
    pub fn ptr() -> Self {
        LLType::Ptr
    }
}

impl Display for LLType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LLType::Int(bits) => write!(f, "i{}", bits),
            LLType::Ptr => write!(f, "ptr"),
            LLType::Struct(s) => write!(f, "{}", s),
        }
    }
}

impl SSAType for LLType {}

/// Struct layout, owned inline. Structural equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LLStruct {
    pub fields: Vec<LLFieldType>,
}

impl LLStruct {
    pub fn new(fields: Vec<LLFieldType>) -> Self {
        LLStruct { fields }
    }

    /// 4×i64 struct representing a BN254 field element in Montgomery form.
    pub fn field_elem() -> Self {
        Self::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ])
    }

    /// 4×i64 struct representing raw (non-Montgomery) limbs.
    pub fn limbs() -> Self {
        Self::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ])
    }

    /// RC header: { Int(64) } — just a refcount.
    pub fn rc_header() -> Self {
        Self::new(vec![LLFieldType::Int(64)])
    }

    /// RC'd fixed-size array: { Inline(RcHeader), Int(64) table_id, InlineArray(elem_struct, count) }
    pub fn rc_array(elem: LLStruct, count: usize) -> Self {
        Self::new(vec![
            LLFieldType::Inline(Self::rc_header()),
            LLFieldType::Int(64),
            LLFieldType::InlineArray(elem, count),
        ])
    }

    // ── AD node structs ────────────────────────────────────────────────
    // All AD nodes share a common prefix: { Inline(RcHeader), Int(32) }
    // so the tag can be read from any node pointer using ad_node_base().

    /// Common prefix for all AD nodes: { RC, tag }.
    pub fn ad_node_base() -> Self {
        Self::new(vec![
            LLFieldType::Inline(Self::rc_header()),
            LLFieldType::Int(32),
        ])
    }

    /// AD constant node: { RC, tag, FieldElem(value) }
    pub fn ad_const_node() -> Self {
        Self::new(vec![
            LLFieldType::Inline(Self::rc_header()),
            LLFieldType::Int(32),
            LLFieldType::Inline(Self::field_elem()),
        ])
    }

    /// AD witness node: { RC, tag, Int(64)(index) }
    pub fn ad_witness_node() -> Self {
        Self::new(vec![
            LLFieldType::Inline(Self::rc_header()),
            LLFieldType::Int(32),
            LLFieldType::Int(64),
        ])
    }

    /// AD sum node: { RC, tag, Ptr(a), Ptr(b), FieldElem(da), FieldElem(db), FieldElem(dc) }
    pub fn ad_sum_node() -> Self {
        Self::new(vec![
            LLFieldType::Inline(Self::rc_header()),
            LLFieldType::Int(32),
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Inline(Self::field_elem()),
            LLFieldType::Inline(Self::field_elem()),
            LLFieldType::Inline(Self::field_elem()),
        ])
    }

    /// AD mul-const node: { RC, tag, FieldElem(coeff), Ptr(value), FieldElem(da), FieldElem(db), FieldElem(dc) }
    pub fn ad_mul_const_node() -> Self {
        Self::new(vec![
            LLFieldType::Inline(Self::rc_header()),
            LLFieldType::Int(32),
            LLFieldType::Inline(Self::field_elem()),
            LLFieldType::Ptr,
            LLFieldType::Inline(Self::field_elem()),
            LLFieldType::Inline(Self::field_elem()),
            LLFieldType::Inline(Self::field_elem()),
        ])
    }

    /// VM struct used by the forward-pass/witgen entrypoint.
    ///
    /// Keep this in field-index order with `mavros-wasm-layout`.
    pub fn witgen_vm() -> Self {
        Self::new(vec![
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Ptr,
            LLFieldType::Int(32),
        ])
    }

    pub const WITGEN_VM_WITNESS: usize = 0;
    pub const WITGEN_VM_A: usize = 1;
    pub const WITGEN_VM_B: usize = 2;
    pub const WITGEN_VM_C: usize = 3;
    /// Cursor into the witness multiplicities section. Mirrors VM
    /// `multiplicities_witness`: each first-use lookup snapshots this into
    /// its own per-table slot, then bumps it by the table's length.
    pub const WITGEN_VM_MULTS_CURSOR: usize = 4;
    pub const WITGEN_VM_LOOKUPS_A: usize = 5;
    pub const WITGEN_VM_LOOKUPS_B: usize = 6;
    pub const WITGEN_VM_LOOKUPS_C: usize = 7;
    pub const WITGEN_VM_INPUTS: usize = 8;
    /// Cursor producing the next free table index. Mirrors `vm.tables.len()`
    /// — first-use lookups snapshot it into their per-table slot, then bump
    /// it by 1.
    pub const WITGEN_VM_NEXT_TABLE_IDX: usize = 9;
    /// Cursor for the constraints tables section. Mirrors VM
    /// `elem_inverses_constraint_section_offset`.
    pub const WITGEN_VM_CURRENT_CNST_TABLES_OFF: usize = 10;
    /// Cursor for the post-commitment witness tables section, relative to
    /// `challenges_start`. Mirrors VM `elem_inverses_witness_section_offset`.
    pub const WITGEN_VM_CURRENT_WIT_TABLES_OFF: usize = 11;
    /// Snapshot slot for the rangecheck-8 table's constraint-table offset.
    pub const WITGEN_VM_RNGCHK_8_CNST_OFF: usize = 12;
    /// Snapshot slot for the rangecheck-8 table's post-commitment witness
    /// table offset.
    pub const WITGEN_VM_RNGCHK_8_WIT_OFF: usize = 13;
    /// Snapshot slot for the rangecheck-8 table's `multiplicities_wit` base.
    /// Sentinel `null` until first rangecheck-8 Lookup; then carries the
    /// snapshotted `mults_cursor`. Mirrors `TableInfo.multiplicities_wit`.
    pub const WITGEN_VM_RNGCHK_8_MULTS_BASE: usize = 14;
    /// Snapshot slot for the rangecheck-8 table's index. Sentinel `u32::MAX`
    /// until first rangecheck-8 Lookup; mirrors `vm.rgchk_8: Option<usize>`.
    pub const WITGEN_VM_RNGCHK_8_TABLE_IDX: usize = 15;

    /// VM struct used by the reverse AD entrypoint.
    ///
    /// Keep this in field-index order with `mavros-wasm-layout`.
    pub fn ad_vm() -> Self {
        Self::new(vec![
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Ptr,
            LLFieldType::Int(32),
            LLFieldType::Ptr,
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
        ])
    }

    pub const AD_VM_OUT_DA: usize = 0;
    pub const AD_VM_OUT_DB: usize = 1;
    pub const AD_VM_OUT_DC: usize = 2;
    pub const AD_VM_COEFFS: usize = 3;
    pub const AD_VM_CURRENT_WIT_OFF: usize = 4;
    pub const AD_VM_COEFFS_BASE: usize = 5;
    pub const AD_VM_CURRENT_LOOKUP_WIT_OFF: usize = 6;
    /// Cursor for the constraints tables section. Mirrors VM
    /// `current_cnst_tables_off`: each first-use lookup snapshots this and
    /// then bumps it by the table's constraint footprint.
    pub const AD_VM_CURRENT_CNST_TABLES_OFF: usize = 7;
    /// Cursor for the witness tables section.
    pub const AD_VM_CURRENT_WIT_TABLES_OFF: usize = 8;
    /// Cursor for the witness multiplicities section.
    pub const AD_VM_CURRENT_WIT_MULTIPLICITIES_OFF: usize = 9;
    /// Snapshot slot for the rangecheck-8 table's `inv_cnst_off`. Sentinel
    /// `u32::MAX` until the first rangecheck-8 DLookup allocates the table.
    pub const AD_VM_RNGCHK_8_INV_CNST_OFF: usize = 10;

    /// AD tag constants.
    pub const AD_TAG_CONST: u64 = 0;
    pub const AD_TAG_WITNESS: u64 = 1;
    pub const AD_TAG_SUM: u64 = 2;
    pub const AD_TAG_MUL_CONST: u64 = 3;

    /// A struct is value-safe if all fields are Int, Ptr, or Inline(value_safe).
    /// Value-safe structs can be used as SSA values (`LLType::Struct`).
    pub fn is_value_safe(&self) -> bool {
        self.fields.iter().all(|f| match f {
            LLFieldType::Int(_) | LLFieldType::Ptr => true,
            LLFieldType::Inline(inner) => inner.is_value_safe(),
            LLFieldType::InlineArray(_, _) | LLFieldType::FlexArray(_) => false,
        })
    }
}

impl Display for LLStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ {} }}",
            self.fields.iter().map(|ft| ft.to_string()).join(", ")
        )
    }
}

/// What a struct field / memory slot holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LLFieldType {
    Int(u32),
    Ptr,
    /// Nested struct embedded in place.
    Inline(LLStruct),
    /// Fixed-count contiguous array of identical structs.
    InlineArray(LLStruct, usize),
    /// Variable-length trailing array (C99 flexible array member).
    FlexArray(LLStruct),
}

impl LLFieldType {
    /// Convert to the corresponding LLType (for Int/Ptr/Inline).
    /// Panics on InlineArray/FlexArray (memory-only).
    pub fn to_ll_type(&self) -> LLType {
        match self {
            LLFieldType::Int(bits) => LLType::Int(*bits),
            LLFieldType::Ptr => LLType::Ptr,
            LLFieldType::Inline(s) => LLType::Struct(s.clone()),
            LLFieldType::InlineArray(_, _) | LLFieldType::FlexArray(_) => {
                panic!("InlineArray/FlexArray fields are memory-only; cannot convert to LLType")
            }
        }
    }
}

impl Display for LLFieldType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LLFieldType::Int(bits) => write!(f, "i{}", bits),
            LLFieldType::Ptr => write!(f, "ptr"),
            LLFieldType::Inline(s) => write!(f, "{}", s),
            LLFieldType::InlineArray(s, count) => write!(f, "[{} x {}]", count, s),
            LLFieldType::FlexArray(s) => write!(f, "[? x {}]", s),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Op-kind enums
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntArithOp {
    Add,
    Sub,
    Mul,
    UDiv,
    URem,
    And,
    Or,
    Xor,
    Shl,
    UShr,
}

impl Display for IntArithOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntArithOp::Add => "add",
            IntArithOp::Sub => "sub",
            IntArithOp::Mul => "mul",
            IntArithOp::UDiv => "udiv",
            IntArithOp::URem => "urem",
            IntArithOp::And => "and",
            IntArithOp::Or => "or",
            IntArithOp::Xor => "xor",
            IntArithOp::Shl => "shl",
            IntArithOp::UShr => "ushr",
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntCmpOp {
    Eq,
    ULt,
    SLt,
}

impl Display for IntCmpOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntCmpOp::Eq => "eq",
            IntCmpOp::ULt => "ult",
            IntCmpOp::SLt => "slt",
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl Display for FieldArithOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = match self {
            FieldArithOp::Add => "field.add",
            FieldArithOp::Sub => "field.sub",
            FieldArithOp::Mul => "field.mul",
            FieldArithOp::Div => "field.div",
        };
        write!(f, "{}", s)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LLOp
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub enum LLOp {
    // ── Constants ────────────────────────────────────────────────────────
    IntConst {
        result: ValueId,
        bits: u32,
        value: u64,
    },
    NullPtr {
        result: ValueId,
    },

    // ── Integer Arithmetic ──────────────────────────────────────────────
    IntArith {
        kind: IntArithOp,
        result: ValueId,
        a: ValueId,
        b: ValueId,
    },
    Not {
        result: ValueId,
        value: ValueId,
    },

    // ── Integer Comparison ──────────────────────────────────────────────
    IntCmp {
        kind: IntCmpOp,
        result: ValueId,
        a: ValueId,
        b: ValueId,
    },

    // ── Width conversion ────────────────────────────────────────────────
    Truncate {
        result: ValueId,
        value: ValueId,
        to_bits: u32,
    },
    ZExt {
        result: ValueId,
        value: ValueId,
        to_bits: u32,
    },

    // ── Field Arithmetic ────────────────────────────────────────────────
    FieldArith {
        kind: FieldArithOp,
        result: ValueId,
        a: ValueId,
        b: ValueId,
    },
    FieldNeg {
        result: ValueId,
        src: ValueId,
    },
    FieldEq {
        result: ValueId,
        a: ValueId,
        b: ValueId,
    },
    FieldToLimbs {
        result: ValueId,
        src: ValueId,
    },
    FieldFromLimbs {
        result: ValueId,
        limbs: ValueId,
    },

    // ── Aggregate ───────────────────────────────────────────────────────
    MkStruct {
        result: ValueId,
        struct_type: LLStruct,
        fields: Vec<ValueId>,
    },
    ExtractField {
        result: ValueId,
        value: ValueId,
        struct_type: LLStruct,
        field: usize,
    },
    // ── Memory ──────────────────────────────────────────────────────────
    HeapAlloc {
        result: ValueId,
        struct_type: LLStruct,
        flex_count: Option<ValueId>,
    },
    Free {
        ptr: ValueId,
    },
    Load {
        result: ValueId,
        ptr: ValueId,
        ty: LLType,
    },
    Store {
        ptr: ValueId,
        value: ValueId,
    },
    StructFieldPtr {
        result: ValueId,
        ptr: ValueId,
        struct_type: LLStruct,
        field: usize,
    },
    ArrayElemPtr {
        result: ValueId,
        ptr: ValueId,
        elem_type: LLStruct,
        index: ValueId,
    },
    Memcpy {
        dst: ValueId,
        src: ValueId,
        struct_type: LLStruct,
        count: Option<ValueId>,
    },

    // ── Selection ───────────────────────────────────────────────────────
    Select {
        result: ValueId,
        cond: ValueId,
        if_t: ValueId,
        if_f: ValueId,
    },

    // ── Calls ───────────────────────────────────────────────────────────
    Call {
        results: Vec<ValueId>,
        func: FunctionId,
        args: Vec<ValueId>,
    },

    // ── Globals ─────────────────────────────────────────────────────────
    GlobalAddr {
        result: ValueId,
        global_id: usize,
    },

    // ── Trap ────────────────────────────────────────────────────────────
    Trap,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Instruction impl for LLOp
// ═══════════════════════════════════════════════════════════════════════════════

impl Instruction for LLOp {
    fn get_inputs(&self) -> impl Iterator<Item = &ValueId> {
        match self {
            // No inputs (constants / traps / etc.)
            LLOp::IntConst { .. } | LLOp::NullPtr { .. } | LLOp::GlobalAddr { .. } | LLOp::Trap => {
                vec![].into_iter()
            }

            // Unary
            LLOp::Not { value, .. }
            | LLOp::FieldNeg { src: value, .. }
            | LLOp::FieldToLimbs { src: value, .. }
            | LLOp::FieldFromLimbs { limbs: value, .. } => vec![value].into_iter(),

            LLOp::Truncate { value, .. } | LLOp::ZExt { value, .. } => vec![value].into_iter(),

            LLOp::Free { ptr } => vec![ptr].into_iter(),

            // Binary
            LLOp::IntArith { a, b, .. }
            | LLOp::IntCmp { a, b, .. }
            | LLOp::FieldArith { a, b, .. }
            | LLOp::FieldEq { a, b, .. } => vec![a, b].into_iter(),

            LLOp::Store { ptr, value } => vec![ptr, value].into_iter(),
            LLOp::Load { ptr, .. } => vec![ptr].into_iter(),

            // Ternary
            LLOp::Select {
                cond, if_t, if_f, ..
            } => vec![cond, if_t, if_f].into_iter(),

            // Struct ops
            LLOp::MkStruct { fields, .. } => fields.iter().collect::<Vec<_>>().into_iter(),
            LLOp::ExtractField { value, .. } => vec![value].into_iter(),

            // Memory with pointer
            LLOp::HeapAlloc { flex_count, .. } => flex_count.iter().collect::<Vec<_>>().into_iter(),
            LLOp::StructFieldPtr { ptr, .. } => vec![ptr].into_iter(),
            LLOp::ArrayElemPtr { ptr, index, .. } => vec![ptr, index].into_iter(),
            LLOp::Memcpy {
                dst, src, count, ..
            } => {
                let mut v = vec![dst, src];
                v.extend(count.iter());
                v.into_iter()
            }

            // Call
            LLOp::Call { args, .. } => args.iter().collect::<Vec<_>>().into_iter(),
        }
    }

    fn get_results(&self) -> impl Iterator<Item = &ValueId> {
        match self {
            // Single result
            LLOp::IntConst { result, .. }
            | LLOp::NullPtr { result }
            | LLOp::IntArith { result, .. }
            | LLOp::Not { result, .. }
            | LLOp::IntCmp { result, .. }
            | LLOp::Truncate { result, .. }
            | LLOp::ZExt { result, .. }
            | LLOp::FieldArith { result, .. }
            | LLOp::FieldNeg { result, .. }
            | LLOp::FieldEq { result, .. }
            | LLOp::FieldToLimbs { result, .. }
            | LLOp::FieldFromLimbs { result, .. }
            | LLOp::MkStruct { result, .. }
            | LLOp::ExtractField { result, .. }
            | LLOp::HeapAlloc { result, .. }
            | LLOp::Load { result, .. }
            | LLOp::StructFieldPtr { result, .. }
            | LLOp::ArrayElemPtr { result, .. }
            | LLOp::Select { result, .. }
            | LLOp::GlobalAddr { result, .. } => vec![result].into_iter(),

            // Multi-result
            LLOp::Call { results, .. } => results.iter().collect::<Vec<_>>().into_iter(),

            // No result
            LLOp::Free { .. } | LLOp::Store { .. } | LLOp::Memcpy { .. } | LLOp::Trap => {
                vec![].into_iter()
            }
        }
    }

    fn get_inputs_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            // No inputs
            LLOp::IntConst { .. } | LLOp::NullPtr { .. } | LLOp::GlobalAddr { .. } | LLOp::Trap => {
                vec![].into_iter()
            }

            // Unary
            LLOp::Not { value, .. }
            | LLOp::FieldNeg { src: value, .. }
            | LLOp::FieldToLimbs { src: value, .. }
            | LLOp::FieldFromLimbs { limbs: value, .. } => vec![value].into_iter(),

            LLOp::Truncate { value, .. } | LLOp::ZExt { value, .. } => vec![value].into_iter(),

            LLOp::Free { ptr } => vec![ptr].into_iter(),

            // Binary
            LLOp::IntArith { a, b, .. }
            | LLOp::IntCmp { a, b, .. }
            | LLOp::FieldArith { a, b, .. }
            | LLOp::FieldEq { a, b, .. } => vec![a, b].into_iter(),

            LLOp::Store { ptr, value } => vec![ptr, value].into_iter(),
            LLOp::Load { ptr, .. } => vec![ptr].into_iter(),

            // Ternary
            LLOp::Select {
                cond, if_t, if_f, ..
            } => vec![cond, if_t, if_f].into_iter(),

            // Struct ops
            LLOp::MkStruct { fields, .. } => fields.iter_mut().collect::<Vec<_>>().into_iter(),
            LLOp::ExtractField { value, .. } => vec![value].into_iter(),

            // Memory with pointer
            LLOp::HeapAlloc { flex_count, .. } => {
                flex_count.iter_mut().collect::<Vec<_>>().into_iter()
            }
            LLOp::StructFieldPtr { ptr, .. } => vec![ptr].into_iter(),
            LLOp::ArrayElemPtr { ptr, index, .. } => vec![ptr, index].into_iter(),
            LLOp::Memcpy {
                dst, src, count, ..
            } => {
                let mut v = vec![dst, src];
                v.extend(count.iter_mut());
                v.into_iter()
            }

            // Call
            LLOp::Call { args, .. } => args.iter_mut().collect::<Vec<_>>().into_iter(),
        }
    }

    fn get_operands_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            LLOp::IntConst { result, .. }
            | LLOp::NullPtr { result }
            | LLOp::GlobalAddr { result, .. } => vec![result].into_iter(),

            LLOp::Trap => vec![].into_iter(),

            LLOp::Not { result, value }
            | LLOp::FieldNeg { result, src: value }
            | LLOp::FieldToLimbs { result, src: value }
            | LLOp::FieldFromLimbs {
                result,
                limbs: value,
            } => vec![result, value].into_iter(),

            LLOp::Truncate { result, value, .. } | LLOp::ZExt { result, value, .. } => {
                vec![result, value].into_iter()
            }

            LLOp::IntArith { result, a, b, .. }
            | LLOp::IntCmp { result, a, b, .. }
            | LLOp::FieldArith { result, a, b, .. }
            | LLOp::FieldEq { result, a, b, .. } => vec![result, a, b].into_iter(),

            LLOp::Select {
                result,
                cond,
                if_t,
                if_f,
            } => vec![result, cond, if_t, if_f].into_iter(),

            LLOp::Free { ptr } => vec![ptr].into_iter(),
            LLOp::Store { ptr, value } => vec![ptr, value].into_iter(),
            LLOp::Load { result, ptr, .. } => vec![result, ptr].into_iter(),

            LLOp::MkStruct { result, fields, .. } => {
                let mut v = vec![result];
                v.extend(fields.iter_mut());
                v.into_iter()
            }
            LLOp::ExtractField { result, value, .. } => vec![result, value].into_iter(),

            LLOp::HeapAlloc {
                result, flex_count, ..
            } => {
                let mut v = vec![result];
                v.extend(flex_count.iter_mut());
                v.into_iter()
            }
            LLOp::StructFieldPtr { result, ptr, .. } => vec![result, ptr].into_iter(),
            LLOp::ArrayElemPtr {
                result, ptr, index, ..
            } => vec![result, ptr, index].into_iter(),
            LLOp::Memcpy {
                dst, src, count, ..
            } => {
                let mut v = vec![dst, src];
                v.extend(count.iter_mut());
                v.into_iter()
            }

            LLOp::Call { results, args, .. } => {
                let mut v: Vec<&mut ValueId> = results.iter_mut().collect();
                v.extend(args.iter_mut());
                v.into_iter()
            }
        }
    }

    fn get_static_call_targets(&self) -> Vec<FunctionId> {
        match self {
            LLOp::Call { func, .. } => vec![*func],
            _ => vec![],
        }
    }

    fn display_instruction(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        annotate_value: &dyn Fn(ValueId) -> String,
    ) -> String {
        let v = |id: ValueId| format!("v{}{}", id.0, annotate_value(id));
        let vr = |id: ValueId| format!("v{}", id.0); // raw, no annotation (for inputs)
        match self {
            LLOp::IntConst {
                result,
                bits,
                value,
            } => {
                format!("{} = int_const i{} {}", v(*result), bits, value)
            }
            LLOp::NullPtr { result } => {
                format!("{} = null_ptr", v(*result))
            }
            LLOp::IntArith { kind, result, a, b } => {
                format!("{} = {} {}, {}", v(*result), kind, vr(*a), vr(*b))
            }
            LLOp::Not { result, value } => {
                format!("{} = not {}", v(*result), vr(*value))
            }
            LLOp::IntCmp { kind, result, a, b } => {
                format!("{} = icmp.{} {}, {}", v(*result), kind, vr(*a), vr(*b))
            }
            LLOp::Truncate {
                result,
                value,
                to_bits,
            } => {
                format!("{} = trunc {} to i{}", v(*result), vr(*value), to_bits)
            }
            LLOp::ZExt {
                result,
                value,
                to_bits,
            } => {
                format!("{} = zext {} to i{}", v(*result), vr(*value), to_bits)
            }
            LLOp::FieldArith { kind, result, a, b } => {
                format!("{} = {} {}, {}", v(*result), kind, vr(*a), vr(*b))
            }
            LLOp::FieldNeg { result, src } => {
                format!("{} = field.neg {}", v(*result), vr(*src))
            }
            LLOp::FieldEq { result, a, b } => {
                format!("{} = field.eq {}, {}", v(*result), vr(*a), vr(*b))
            }
            LLOp::FieldToLimbs { result, src } => {
                format!("{} = field.to_limbs {}", v(*result), vr(*src))
            }
            LLOp::FieldFromLimbs { result, limbs } => {
                format!("{} = field.from_limbs {}", v(*result), vr(*limbs))
            }
            LLOp::MkStruct {
                result,
                struct_type,
                fields,
            } => {
                let fields_str = fields.iter().map(|f| vr(*f)).join(", ");
                format!(
                    "{} = mk_struct {} {{ {} }}",
                    v(*result),
                    struct_type,
                    fields_str
                )
            }
            LLOp::ExtractField {
                result,
                value,
                struct_type,
                field,
            } => {
                format!(
                    "{} = extract_field {}, {}, {}",
                    v(*result),
                    vr(*value),
                    struct_type,
                    field
                )
            }
            LLOp::HeapAlloc {
                result,
                struct_type,
                flex_count,
            } => {
                let flex_str = match flex_count {
                    Some(c) => format!(", flex_count={}", vr(*c)),
                    None => "".to_string(),
                };
                format!("{} = heap_alloc {}{}", v(*result), struct_type, flex_str)
            }
            LLOp::Free { ptr } => {
                format!("free {}", vr(*ptr))
            }
            LLOp::Load { result, ptr, ty } => {
                format!("{} = load {}, {}", v(*result), vr(*ptr), ty)
            }
            LLOp::Store { ptr, value } => {
                format!("store {}, {}", vr(*ptr), vr(*value))
            }
            LLOp::StructFieldPtr {
                result,
                ptr,
                struct_type,
                field,
            } => {
                format!(
                    "{} = struct_field_ptr {}, {}, {}",
                    v(*result),
                    vr(*ptr),
                    struct_type,
                    field
                )
            }
            LLOp::ArrayElemPtr {
                result,
                ptr,
                elem_type,
                index,
            } => {
                format!(
                    "{} = array_elem_ptr {}, {}, {}",
                    v(*result),
                    vr(*ptr),
                    elem_type,
                    vr(*index)
                )
            }
            LLOp::Memcpy {
                dst,
                src,
                struct_type,
                count,
            } => {
                let count_str = match count {
                    Some(c) => format!(", count={}", vr(*c)),
                    None => "".to_string(),
                };
                format!(
                    "memcpy {}, {}, {}{}",
                    vr(*dst),
                    vr(*src),
                    struct_type,
                    count_str
                )
            }
            LLOp::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                format!(
                    "{} = select {}, {}, {}",
                    v(*result),
                    vr(*cond),
                    vr(*if_t),
                    vr(*if_f)
                )
            }
            LLOp::Call {
                results,
                func,
                args,
            } => {
                let results_str = results.iter().map(|r| v(*r)).join(", ");
                let args_str = args.iter().map(|a| vr(*a)).join(", ");
                format!(
                    "{} = call {}@{}({})",
                    results_str,
                    func_name(*func),
                    func.0,
                    args_str
                )
            }
            LLOp::GlobalAddr { result, global_id } => {
                format!("{} = global_addr g{}", v(*result), global_id)
            }
            LLOp::Trap => "trap".to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Type aliases
// ═══════════════════════════════════════════════════════════════════════════════

pub type LLSSA = SSA<LLOp, LLType>;
pub type LLFunction = Function<LLOp, LLType>;
pub type LLBlock = Block<LLOp, LLType>;

// Re-export DMatrix for use by other LLSSA modules.
pub use crate::compiler::ssa::DMatrix as LLDMatrix;

// Builder methods are provided by the LLEmitter trait in block_builder.rs.

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::block_builder::{LLBlockEmitter, LLEmitter};
    use crate::compiler::ssa::DefaultSsaAnnotator;

    #[test]
    fn field_elem_is_value_safe() {
        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);
        assert!(field_elem.is_value_safe());
    }

    #[test]
    fn flex_array_is_memory_only() {
        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);
        let slice_header = LLStruct::new(vec![LLFieldType::Int(64), LLFieldType::Int(32)]);
        let dyn_slice = LLStruct::new(vec![
            LLFieldType::Inline(slice_header),
            LLFieldType::FlexArray(field_elem),
        ]);
        assert!(!dyn_slice.is_value_safe());
    }

    #[test]
    fn inline_array_is_memory_only() {
        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);
        let rc_header = LLStruct::new(vec![LLFieldType::Int(64)]);
        let rc_array = LLStruct::new(vec![
            LLFieldType::Inline(rc_header),
            LLFieldType::InlineArray(field_elem, 5),
        ]);
        assert!(!rc_array.is_value_safe());
    }

    #[test]
    fn nested_inline_is_value_safe() {
        let inner = LLStruct::new(vec![LLFieldType::Int(64), LLFieldType::Ptr]);
        let outer = LLStruct::new(vec![LLFieldType::Inline(inner), LLFieldType::Int(32)]);
        assert!(outer.is_value_safe());
    }

    #[test]
    fn display_types() {
        assert_eq!(LLType::i1().to_string(), "i1");
        assert_eq!(LLType::i32().to_string(), "i32");
        assert_eq!(LLType::i64().to_string(), "i64");
        assert_eq!(LLType::ptr().to_string(), "ptr");

        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);
        assert_eq!(field_elem.to_string(), "{ i64, i64, i64, i64 }");
        assert_eq!(
            LLType::Struct(field_elem.clone()).to_string(),
            "{ i64, i64, i64, i64 }"
        );

        let arr_field = LLFieldType::InlineArray(field_elem.clone(), 5);
        assert_eq!(arr_field.to_string(), "[5 x { i64, i64, i64, i64 }]");

        let flex_field = LLFieldType::FlexArray(field_elem);
        assert_eq!(flex_field.to_string(), "[? x { i64, i64, i64, i64 }]");
    }

    #[test]
    fn field_type_to_ll_type() {
        assert_eq!(LLFieldType::Int(32).to_ll_type(), LLType::i32());
        assert_eq!(LLFieldType::Ptr.to_ll_type(), LLType::ptr());
        let s = LLStruct::new(vec![LLFieldType::Int(64)]);
        assert_eq!(
            LLFieldType::Inline(s.clone()).to_ll_type(),
            LLType::Struct(s)
        );
    }

    #[test]
    #[should_panic]
    fn inline_array_to_ll_type_panics() {
        let s = LLStruct::new(vec![LLFieldType::Int(64)]);
        let _ = LLFieldType::InlineArray(s, 3).to_ll_type();
    }

    #[test]
    fn build_simple_function() {
        let mut ssa = LLSSA::with_main("test_main".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let x = e.int_const(64, 42);
            let y = e.int_const(64, 7);
            let z = e.int_add(x, y);
            e.terminate_return(vec![z]);
        }

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("int_const i64 42"));
        assert!(dump.contains("int_const i64 7"));
        assert!(dump.contains("add"));
        assert!(dump.contains("return"));
    }

    #[test]
    fn build_struct_ops() {
        let mut ssa = LLSSA::with_main("struct_test".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let l0 = e.int_const(64, 1);
            let l1 = e.int_const(64, 0);
            let l2 = e.int_const(64, 0);
            let l3 = e.int_const(64, 0);
            let s = e.mk_struct(field_elem.clone(), vec![l0, l1, l2, l3]);
            let f0 = e.extract_field(s, field_elem, 0);
            e.terminate_return(vec![f0, s]);
        }

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("mk_struct"));
        assert!(dump.contains("extract_field"));
    }

    #[test]
    fn build_memory_ops() {
        let mut ssa = LLSSA::with_main("memory_test".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        let rc_header = LLStruct::new(vec![LLFieldType::Int(64)]);
        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);
        let rc_array = LLStruct::new(vec![
            LLFieldType::Inline(rc_header.clone()),
            LLFieldType::InlineArray(field_elem.clone(), 3),
        ]);

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let arr = e.heap_alloc(rc_array.clone(), None);
            let rc_ptr = e.struct_field_ptr(arr, rc_array.clone(), 0);
            let rc_word = e.struct_field_ptr(rc_ptr, rc_header, 0);
            let one = e.int_const(64, 1);
            e.ll_store(rc_word, one);

            let data = e.struct_field_ptr(arr, rc_array, 1);
            let idx = e.int_const(64, 0);
            let elem_ptr = e.array_elem_ptr(data, field_elem, idx);
            let loaded = e.ll_load(elem_ptr, LLType::i64());
            e.free(arr);
            e.terminate_return(vec![loaded]);
        }

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("heap_alloc"));
        assert!(dump.contains("struct_field_ptr"));
        assert!(dump.contains("array_elem_ptr"));
        assert!(dump.contains("store"));
        assert!(dump.contains("load"));
        assert!(dump.contains("free"));
    }

    #[test]
    fn build_call_and_select() {
        let mut ssa = LLSSA::with_main("call_test".to_string());
        let helper_id = ssa.add_function("helper".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let a = e.int_const(64, 1);
            let b = e.int_const(64, 2);
            let results = e.call(helper_id, vec![a, b], 1);
            let cond = e.int_eq(results[0], a);
            let selected = e.select(cond, a, b);
            e.terminate_return(vec![selected]);
        }

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("call helper@"));
        assert!(dump.contains("icmp.eq"));
        assert!(dump.contains("select"));
    }

    #[test]
    fn build_field_ops() {
        let mut ssa = LLSSA::with_main("field_test".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let l0 = e.int_const(64, 1);
            let l1 = e.int_const(64, 0);
            let l2 = e.int_const(64, 0);
            let l3 = e.int_const(64, 0);
            let a = e.mk_struct(field_elem.clone(), vec![l0, l1, l2, l3]);
            let b = e.mk_struct(field_elem, vec![l0, l1, l2, l3]);

            let c = e.field_arith(FieldArithOp::Add, a, b);
            let d = e.field_neg(c);
            let eq = e.field_eq(c, d);
            let limbs = e.field_to_limbs(d);
            let back = e.field_from_limbs(limbs);
            e.terminate_return(vec![eq, back]);
        }

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("field.add"));
        assert!(dump.contains("field.neg"));
        assert!(dump.contains("field.eq"));
        assert!(dump.contains("field.to_limbs"));
        assert!(dump.contains("field.from_limbs"));
    }

    #[test]
    fn build_width_and_global() {
        let mut ssa = LLSSA::with_main("width_test".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let x = e.int_const(64, 256);
            let narrow = e.truncate(x, 8);
            let wide = e.zext(narrow, 64);
            let gp = e.global_addr(3);
            e.ll_store(gp, wide);
            e.trap();
        }

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("trunc"));
        assert!(dump.contains("zext"));
        assert!(dump.contains("global_addr g3"));
        assert!(dump.contains("trap"));
    }

    #[test]
    fn build_branching() {
        let mut ssa = LLSSA::with_main("branch_test".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();
        let then_blk = func.add_block();
        let else_blk = func.add_block();
        let merge_blk = func.add_block();

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let x = e.int_const(64, 42);
            let zero = e.int_const(64, 0);
            let cond = e.int_eq(x, zero);
            e.terminate_jmp_if(cond, then_blk, else_blk);
        }
        {
            let mut e = LLBlockEmitter::new(func, then_blk);
            let one = e.int_const(64, 1);
            e.terminate_jmp(merge_blk, vec![one]);
        }
        {
            let mut e = LLBlockEmitter::new(func, else_blk);
            let two = e.int_const(64, 2);
            e.terminate_jmp(merge_blk, vec![two]);
        }

        func.terminate_block_with_return(merge_blk, vec![]);

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("jmp_if"));
        assert!(dump.contains("jmp block_"));
    }

    #[test]
    fn build_memcpy() {
        let mut ssa = LLSSA::with_main("memcpy_test".to_string());
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        let elem = LLStruct::new(vec![LLFieldType::Int(64)]);

        {
            let mut e = LLBlockEmitter::new(func, entry);
            let dst = e.null_ptr();
            let src = e.null_ptr();
            let count = e.int_const(64, 10);
            e.memcpy(dst, src, elem, Some(count));
            e.terminate_return(vec![]);
        }

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("memcpy"));
        assert!(dump.contains("count="));
    }
}
