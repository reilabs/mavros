//! The low-level SSA representation used in the compiler and its associated types.

pub mod builder;
pub mod type_system;

use itertools::Itertools;
use std::fmt::{self, Display, Formatter};
use std::mem::size_of_val;

use crate::compiler::ssa::{Block, Function, FunctionId, Instruction, SSA, SSAConstants, ValueId};
pub use type_system::Type;

// Re-export DMatrix for use by other LLSSA modules.
pub use super::hlssa::DMatrix;

// LLSSA
// ================================================================================================

/// The low-level SSA exposes runtime details: explicit struct layouts, pointer arithmetic,
/// integer/field arithmetic split, and explicit memory management.
pub type LLSSA = SSA<LLOp, Type, Constant>;

// CONSTANTS
// ================================================================================================

/// The constant used to signal that an object is immortal for the purposes of reference counting.
///
/// The word used to track the refcount for that object should be set to this value, and every
/// refcount operation should check for this value before attempting to modify the refcount.
pub const RC_IMMORTAL_OBJECT: u64 = u64::MAX;

/// The size of the refcount in bytes.
pub const RC_SIZE_BYTES: usize = 8;

/// The size of the refcount in bits.
pub const RC_SIZE_BITS: usize = RC_SIZE_BYTES * 8;

const _: () = assert!(
    size_of_val(&RC_IMMORTAL_OBJECT) == RC_SIZE_BYTES,
    "Size of the immortal refcount constant does not match the size of the refcount value"
);

// CONSTANT STORAGE
// ================================================================================================

/// The value type stored in the low-level SSA's constants table.
///
/// Most LLSSA constants are scalar leaves: integers of a given bit width and the null pointer.
/// Aggregate constants (e.g. the four-limb field element struct) are representable via `Struct`,
/// which pairs a struct layout with one constant value per field.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    /// A scalar integer constant of `bits` bits with the provided `value`.
    Int { bits: u32, value: u128 },

    /// A null pointer constant.
    NullPtr,

    /// An aggregate (used for tuples and other aggregates).
    Struct { layout: LLStruct, values: Vec<Constant> },
}

impl Constant {
    /// True if this constant can legally fill a slot of `field` type.
    ///
    /// `InlineArray`/`FlexArray` fields are memory-only and have no constant form,
    /// so they never match; any other mismatched pairing is rejected too.
    fn matches_field(&self, field: &LLFieldType) -> bool {
        match (self, field) {
            (Constant::Int { bits, .. }, LLFieldType::Int(w)) => bits == w,
            (Constant::NullPtr, LLFieldType::Ptr) => true,
            (Constant::Struct { layout, values }, LLFieldType::Inline(inner)) => {
                layout == inner && layout.accepts(values)
            }
            _ => false,
        }
    }
}

/// The constants table type for LLSSA.
pub type LLSSAConstants = SSAConstants<Constant>;

// LLSSA OPCODES
// ================================================================================================

#[derive(Clone, Debug)]
pub enum LLOp {
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
    Spread {
        result: ValueId,
        value: ValueId,
        bits: u8,
        result_bits: u32,
    },
    Unspread {
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
        bits: u8,
        odd_bits: u32,
        even_bits: u32,
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
        ty: Type,
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

impl Instruction for LLOp {
    fn get_inputs(&self) -> impl Iterator<Item = &ValueId> {
        match self {
            // No inputs (globals / traps / etc.)
            LLOp::GlobalAddr { .. } | LLOp::Trap => vec![].into_iter(),

            // Unary
            LLOp::Not { value, .. }
            | LLOp::FieldNeg { src: value, .. }
            | LLOp::FieldToLimbs { src: value, .. }
            | LLOp::FieldFromLimbs { limbs: value, .. }
            | LLOp::Spread { value, .. }
            | LLOp::Unspread { value, .. } => vec![value].into_iter(),

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
            LLOp::IntArith { result, .. }
            | LLOp::Not { result, .. }
            | LLOp::Spread { result, .. }
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
            LLOp::Unspread {
                result_odd,
                result_even,
                ..
            } => vec![result_odd, result_even].into_iter(),

            // No result
            LLOp::Free { .. } | LLOp::Store { .. } | LLOp::Memcpy { .. } | LLOp::Trap => {
                vec![].into_iter()
            }
        }
    }

    fn get_results_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            LLOp::IntArith { result, .. }
            | LLOp::Not { result, .. }
            | LLOp::Spread { result, .. }
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
            LLOp::Call { results, .. } => results.iter_mut().collect::<Vec<_>>().into_iter(),
            LLOp::Unspread {
                result_odd,
                result_even,
                ..
            } => vec![result_odd, result_even].into_iter(),
            LLOp::Free { .. } | LLOp::Store { .. } | LLOp::Memcpy { .. } | LLOp::Trap => {
                vec![].into_iter()
            }
        }
    }

    fn get_inputs_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            // No inputs
            LLOp::GlobalAddr { .. } | LLOp::Trap => vec![].into_iter(),

            // Unary
            LLOp::Not { value, .. }
            | LLOp::FieldNeg { src: value, .. }
            | LLOp::FieldToLimbs { src: value, .. }
            | LLOp::FieldFromLimbs { limbs: value, .. }
            | LLOp::Spread { value, .. }
            | LLOp::Unspread { value, .. } => vec![value].into_iter(),

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
            LLOp::GlobalAddr { result, .. } => vec![result].into_iter(),

            LLOp::Trap => vec![].into_iter(),

            LLOp::Not { result, value }
            | LLOp::Spread { result, value, .. }
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

            LLOp::Unspread {
                result_odd,
                result_even,
                value,
                ..
            } => vec![result_odd, result_even, value].into_iter(),

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
            LLOp::IntArith { kind, result, a, b } => {
                format!("{} = {} {}, {}", v(*result), kind, vr(*a), vr(*b))
            }
            LLOp::Not { result, value } => {
                format!("{} = not {}", v(*result), vr(*value))
            }
            LLOp::Spread {
                result,
                value,
                bits,
                result_bits,
            } => {
                format!(
                    "{} = spread({}) {} to i{}",
                    v(*result),
                    bits,
                    vr(*value),
                    result_bits
                )
            }
            LLOp::Unspread {
                result_odd,
                result_even,
                value,
                bits,
                odd_bits,
                even_bits,
            } => {
                format!(
                    "{}, {} = unspread({}) {} to i{}, i{}",
                    v(*result_odd),
                    v(*result_even),
                    bits,
                    vr(*value),
                    odd_bits,
                    even_bits
                )
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

// LLSSA TYPE ALIASES
// ================================================================================================

pub type LLFunction = Function<LLOp, Type>;
pub type LLBlock = Block<LLOp, Type>;

// Builder methods are provided by the LLEmitter trait in `block_builder` (sibling module).

// STRUCT LAYOUT
// ================================================================================================

/// Struct layout, owned inline. Structural equality.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        Self::new(vec![LLFieldType::Int(RC_SIZE_BITS as u32)])
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

    /// Per-slot table-info record. Mirrors the byte layout of
    /// `TABLE_INFO_*_OFFSET` in `mavros-wasm-layout`. One of these lives in
    /// each host-visible runtime table slot of the witgen VM struct.
    ///
    /// Field index → name (use `TABLE_INFO_*` constants below):
    ///   0: mults_base   (ptr)
    ///   1: inv_cnst_off (i32)
    ///   2: inv_wit_off  (i32)
    ///   3: num_indices  (i32)
    ///   4: num_values   (i32)
    ///   5: length       (i32)
    pub fn table_info_slot() -> Self {
        Self::new(vec![
            LLFieldType::Ptr,
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
            LLFieldType::Int(32),
        ])
    }

    pub const TABLE_INFO_MULTS_BASE: usize = 0;
    pub const TABLE_INFO_INV_CNST_OFF: usize = 1;
    pub const TABLE_INFO_INV_WIT_OFF: usize = 2;
    pub const TABLE_INFO_NUM_INDICES: usize = 3;
    pub const TABLE_INFO_NUM_VALUES: usize = 4;
    pub const TABLE_INFO_LENGTH: usize = 5;

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
            LLFieldType::Ptr,     // 0  witness
            LLFieldType::Ptr,     // 1  a
            LLFieldType::Ptr,     // 2  a_base
            LLFieldType::Ptr,     // 3  b
            LLFieldType::Ptr,     // 4  c
            LLFieldType::Ptr,     // 5  mults_cursor
            LLFieldType::Ptr,     // 6  lookups_a
            LLFieldType::Ptr,     // 7  lookups_b
            LLFieldType::Ptr,     // 8  lookups_c
            LLFieldType::Ptr,     // 9  inputs
            LLFieldType::Int(32), // 10 tables_len
            LLFieldType::Int(32), // 11 tables_cap
            LLFieldType::Ptr,     // 12 tables_ptr
            LLFieldType::Int(32), // 13 current_cnst_tables_off
            LLFieldType::Int(32), // 14 current_wit_tables_off
            LLFieldType::Int(32), // 15 reserved padding
        ])
    }

    pub const WITGEN_VM_WITNESS: usize = 0;
    pub const WITGEN_VM_A: usize = 1;
    pub const WITGEN_VM_A_BASE: usize = 2;
    pub const WITGEN_VM_B: usize = 3;
    pub const WITGEN_VM_C: usize = 4;
    /// Cursor into the witness multiplicities section. Mirrors VM
    /// `multiplicities_witness`: each first-use lookup snapshots this into
    /// its slot, then bumps it by the table's length.
    pub const WITGEN_VM_MULTS_CURSOR: usize = 5;
    pub const WITGEN_VM_LOOKUPS_A: usize = 6;
    pub const WITGEN_VM_LOOKUPS_B: usize = 7;
    pub const WITGEN_VM_LOOKUPS_C: usize = 8;
    pub const WITGEN_VM_INPUTS: usize = 9;
    /// Cursor counting allocated tables. Each first-use claim assigns the
    /// current `tables_len` as the new table id, then bumps it.
    /// Mirrors `vm.tables.len()`.
    pub const WITGEN_VM_TABLES_LEN: usize = 10;
    /// Capacity of the host-allocated table-info buffer.
    pub const WITGEN_VM_TABLES_CAP: usize = 11;
    /// Base pointer to the host-allocated table-info buffer.
    pub const WITGEN_VM_TABLES_PTR: usize = 12;
    /// Cursor into the constraints-region tables section (advances on
    /// first-use claims by each table's footprint). Forward-side, written-
    /// only — read by Phase 2 via the per-slot `inv_cnst_off` snapshot.
    pub const WITGEN_VM_CURRENT_CNST_TABLES_OFF: usize = 13;
    /// Cursor into the post-commitment witness tables section (relative to
    /// `challenges_start`).
    pub const WITGEN_VM_CURRENT_WIT_TABLES_OFF: usize = 14;

    /// VM struct used by the reverse AD entrypoint.
    ///
    /// Keep this in field-index order with `mavros-wasm-layout`.
    pub fn ad_vm() -> Self {
        Self::new(vec![
            LLFieldType::Ptr,     // 0  out_da
            LLFieldType::Ptr,     // 1  out_db
            LLFieldType::Ptr,     // 2  out_dc
            LLFieldType::Ptr,     // 3  coeffs (cursor)
            LLFieldType::Int(32), // 4  current_wit_off
            LLFieldType::Ptr,     // 5  coeffs_base
            LLFieldType::Int(32), // 6  current_lookup_wit_off
            LLFieldType::Int(32), // 7  current_cnst_tables_off
            LLFieldType::Int(32), // 8  current_wit_tables_off
            LLFieldType::Int(32), // 9  current_wit_multiplicities_off
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
    /// `current_cnst_tables_off`.
    pub const AD_VM_CURRENT_CNST_TABLES_OFF: usize = 7;
    /// Cursor for the witness tables section.
    pub const AD_VM_CURRENT_WIT_TABLES_OFF: usize = 8;
    /// Cursor for the witness multiplicities section.
    pub const AD_VM_CURRENT_WIT_MULTIPLICITIES_OFF: usize = 9;

    /// AD tag constants.
    pub const AD_TAG_CONST: u64 = 0;
    pub const AD_TAG_WITNESS: u64 = 1;
    pub const AD_TAG_SUM: u64 = 2;
    pub const AD_TAG_MUL_CONST: u64 = 3;

    /// A struct is value-safe if all fields are Int, Ptr, or Inline(value_safe).
    /// Value-safe structs can be used as SSA values (`Type::Struct`).
    pub fn is_value_safe(&self) -> bool {
        self.fields.iter().all(|f| match f {
            LLFieldType::Int(_) | LLFieldType::Ptr => true,
            LLFieldType::Inline(inner) => inner.is_value_safe(),
            LLFieldType::InlineArray(_, _) | LLFieldType::FlexArray(_) => false,
        })
    }

    /// True if `values` is exactly one constant per field, in declaration order,
    /// with each constant compatible with its field type (see `Constant::matches_field`).
    pub fn accepts(&self, values: &[Constant]) -> bool {
        self.fields.len() == values.len()
            && self
                .fields
                .iter()
                .zip(values)
                .all(|(f, v)| v.matches_field(f))
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

// FIELD TYPE
// ================================================================================================

/// What a struct field / memory slot holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// Convert to the corresponding `Type` (for Int/Ptr/Inline).
    /// Panics on InlineArray/FlexArray (memory-only).
    pub fn to_ll_type(&self) -> Type {
        match self {
            LLFieldType::Int(bits) => Type::Int(*bits),
            LLFieldType::Ptr => Type::Ptr,
            LLFieldType::Inline(s) => Type::Struct(s.clone()),
            LLFieldType::InlineArray(_, _) | LLFieldType::FlexArray(_) => {
                panic!("InlineArray/FlexArray fields are memory-only; cannot convert to Type")
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

// INT ARITH OPERATION KIND
// ================================================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntArithOp {
    Add,
    Sub,
    Mul,
    UDiv,
    URem,
    SDiv,
    SRem,
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
            IntArithOp::SDiv => "sdiv",
            IntArithOp::SRem => "srem",
            IntArithOp::And => "and",
            IntArithOp::Or => "or",
            IntArithOp::Xor => "xor",
            IntArithOp::Shl => "shl",
            IntArithOp::UShr => "ushr",
        };
        write!(f, "{}", s)
    }
}

// INT COMPARISON KIND
// ================================================================================================

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

// FIELD ARITH OPERATION KIND
// ================================================================================================

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

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::DefaultSSAAnnotator;
    use crate::compiler::ssa::llssa::builder::{LLEmitter, LLSSABuilder};

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
    fn accepts_matching_field_elem() {
        let values: Vec<Constant> = (0..4)
            .map(|v| Constant::Int { bits: 64, value: v })
            .collect();
        assert!(LLStruct::field_elem().accepts(&values));
    }

    #[test]
    fn accepts_rejects_mismatches() {
        let field_elem = LLStruct::field_elem();

        // Wrong arity: only three values for a four-field struct.
        let three: Vec<Constant> = (0..3)
            .map(|v| Constant::Int { bits: 64, value: v })
            .collect();
        assert!(!field_elem.accepts(&three));

        // Wrong int width: i32 where i64 is expected.
        let bad_width = vec![
            Constant::Int { bits: 32, value: 0 },
            Constant::Int { bits: 64, value: 0 },
            Constant::Int { bits: 64, value: 0 },
            Constant::Int { bits: 64, value: 0 },
        ];
        assert!(!field_elem.accepts(&bad_width));

        // Wrong kind: NullPtr where an Int is expected.
        let one_int = LLStruct::new(vec![LLFieldType::Int(64)]);
        assert!(!one_int.accepts(&[Constant::NullPtr]));

        // Ptr field accepts NullPtr.
        let one_ptr = LLStruct::new(vec![LLFieldType::Ptr]);
        assert!(one_ptr.accepts(&[Constant::NullPtr]));

        // Inline field: nested struct constant must match the inner layout.
        let outer = LLStruct::new(vec![LLFieldType::Inline(one_int.clone())]);
        let good_nested = vec![Constant::Struct {
            layout: one_int.clone(),
            values: vec![Constant::Int { bits: 64, value: 7 }],
        }];
        assert!(outer.accepts(&good_nested));
        let bad_nested = vec![Constant::Struct {
            layout: LLStruct::new(vec![LLFieldType::Int(32)]),
            values: vec![Constant::Int { bits: 32, value: 7 }],
        }];
        assert!(!outer.accepts(&bad_nested));

        // InlineArray / FlexArray fields have no constant form.
        let arr = LLStruct::new(vec![LLFieldType::InlineArray(one_int, 1)]);
        assert!(!arr.accepts(&[Constant::Int { bits: 64, value: 0 }]));
    }

    #[test]
    #[should_panic(expected = "incompatible with layout")]
    fn emit_struct_const_rejects_incompatible() {
        let mut ssa = LLSSA::with_main("bad_struct".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            // Only three values for a four-limb field-element layout.
            let values = vec![
                Constant::Int { bits: 64, value: 0 },
                Constant::Int { bits: 64, value: 0 },
                Constant::Int { bits: 64, value: 0 },
            ];
            e.emit_struct_const(LLStruct::field_elem(), values);
        });
    }

    #[test]
    fn display_types() {
        assert_eq!(Type::i1().to_string(), "i1");
        assert_eq!(Type::i32().to_string(), "i32");
        assert_eq!(Type::i64().to_string(), "i64");
        assert_eq!(Type::ptr().to_string(), "ptr");

        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);
        assert_eq!(field_elem.to_string(), "{ i64, i64, i64, i64 }");
        assert_eq!(
            Type::Struct(field_elem.clone()).to_string(),
            "{ i64, i64, i64, i64 }"
        );

        let arr_field = LLFieldType::InlineArray(field_elem.clone(), 5);
        assert_eq!(arr_field.to_string(), "[5 x { i64, i64, i64, i64 }]");

        let flex_field = LLFieldType::FlexArray(field_elem);
        assert_eq!(flex_field.to_string(), "[? x { i64, i64, i64, i64 }]");
    }

    #[test]
    fn field_type_to_ll_type() {
        assert_eq!(LLFieldType::Int(32).to_ll_type(), Type::i32());
        assert_eq!(LLFieldType::Ptr.to_ll_type(), Type::ptr());
        let s = LLStruct::new(vec![LLFieldType::Int(64)]);
        assert_eq!(LLFieldType::Inline(s.clone()).to_ll_type(), Type::Struct(s));
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
        let main_id = ssa.get_main_id();
        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            let x = e.emit_int_const(64, 42);
            let y = e.emit_int_const(64, 7);
            let z = e.int_add(x, y);
            e.terminate_return(vec![z]);
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
        assert!(dump.contains("constants:"));
        assert!(dump.contains("Int { bits: 64, value: 42 }"));
        assert!(dump.contains("Int { bits: 64, value: 7 }"));
        assert!(dump.contains("add"));
        assert!(dump.contains("return"));
    }

    #[test]
    fn build_struct_ops() {
        let mut ssa = LLSSA::with_main("struct_test".to_string());
        let main_id = ssa.get_main_id();

        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);

        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            let l0 = e.emit_int_const(64, 1);
            let l1 = e.emit_int_const(64, 0);
            let l2 = e.emit_int_const(64, 0);
            let l3 = e.emit_int_const(64, 0);
            let s = e.mk_struct(field_elem.clone(), vec![l0, l1, l2, l3]);
            let f0 = e.extract_field(s, field_elem, 0);
            e.terminate_return(vec![f0, s]);
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
        assert!(dump.contains("mk_struct"));
        assert!(dump.contains("extract_field"));
    }

    #[test]
    fn build_memory_ops() {
        let mut ssa = LLSSA::with_main("memory_test".to_string());
        let main_id = ssa.get_main_id();

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

        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            let arr = e.heap_alloc(rc_array.clone(), None);
            let rc_ptr = e.struct_field_ptr(arr, rc_array.clone(), 0);
            let rc_word = e.struct_field_ptr(rc_ptr, rc_header, 0);
            let one = e.emit_int_const(64, 1);
            e.ll_store(rc_word, one);

            let data = e.struct_field_ptr(arr, rc_array, 1);
            let idx = e.emit_int_const(64, 0);
            let elem_ptr = e.array_elem_ptr(data, field_elem, idx);
            let loaded = e.ll_load(elem_ptr, Type::i64());
            e.free(arr);
            e.terminate_return(vec![loaded]);
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
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
        let main_id = ssa.get_main_id();
        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            let a = e.emit_int_const(64, 1);
            let b = e.emit_int_const(64, 2);
            let results = e.call(helper_id, vec![a, b], 1);
            let cond = e.int_eq(results[0], a);
            let selected = e.select(cond, a, b);
            e.terminate_return(vec![selected]);
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
        assert!(dump.contains("call helper@"));
        assert!(dump.contains("icmp.eq"));
        assert!(dump.contains("select"));
    }

    #[test]
    fn build_field_ops() {
        let mut ssa = LLSSA::with_main("field_test".to_string());
        let main_id = ssa.get_main_id();

        let field_elem = LLStruct::new(vec![
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
            LLFieldType::Int(64),
        ]);

        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            let l0 = e.emit_int_const(64, 1);
            let l1 = e.emit_int_const(64, 0);
            let l2 = e.emit_int_const(64, 0);
            let l3 = e.emit_int_const(64, 0);
            let a = e.mk_struct(field_elem.clone(), vec![l0, l1, l2, l3]);
            let b = e.mk_struct(field_elem, vec![l0, l1, l2, l3]);

            let c = e.field_arith(FieldArithOp::Add, a, b);
            let d = e.field_neg(c);
            let eq = e.field_eq(c, d);
            let limbs = e.field_to_limbs(d);
            let back = e.field_from_limbs(limbs);
            e.terminate_return(vec![eq, back]);
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
        assert!(dump.contains("field.add"));
        assert!(dump.contains("field.neg"));
        assert!(dump.contains("field.eq"));
        assert!(dump.contains("field.to_limbs"));
        assert!(dump.contains("field.from_limbs"));
    }

    #[test]
    fn build_width_and_global() {
        let mut ssa = LLSSA::with_main("width_test".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            let x = e.emit_int_const(64, 256);
            let narrow = e.truncate(x, 8);
            let wide = e.zext(narrow, 64);
            let gp = e.global_addr(3);
            e.ll_store(gp, wide);
            e.trap();
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
        assert!(dump.contains("trunc"));
        assert!(dump.contains("zext"));
        assert!(dump.contains("global_addr g3"));
        assert!(dump.contains("trap"));
    }

    #[test]
    fn build_branching() {
        let mut ssa = LLSSA::with_main("branch_test".to_string());
        let main_id = ssa.get_main_id();
        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let then_blk = fb.function.add_block();
            let else_blk = fb.function.add_block();
            let merge_blk = fb.function.add_block();

            {
                let mut e = fb.block(entry);
                let x = e.emit_int_const(64, 42);
                let zero = e.emit_int_const(64, 0);
                let cond = e.int_eq(x, zero);
                e.terminate_jmp_if(cond, then_blk, else_blk);
            }
            {
                let mut e = fb.block(then_blk);
                let one = e.emit_int_const(64, 1);
                e.terminate_jmp(merge_blk, vec![one]);
            }
            {
                let mut e = fb.block(else_blk);
                let two = e.emit_int_const(64, 2);
                e.terminate_jmp(merge_blk, vec![two]);
            }

            fb.function.terminate_block_with_return(merge_blk, vec![]);
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
        assert!(dump.contains("jmp_if"));
        assert!(dump.contains("jmp block_"));
    }

    #[test]
    fn build_memcpy() {
        let mut ssa = LLSSA::with_main("memcpy_test".to_string());
        let main_id = ssa.get_main_id();

        let elem = LLStruct::new(vec![LLFieldType::Int(64)]);

        let mut sb = LLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let mut e = fb.block(entry);
            let dst = e.emit_nullptr_const();
            let src = e.emit_nullptr_const();
            let count = e.emit_int_const(64, 10);
            e.memcpy(dst, src, elem, Some(count));
            e.terminate_return(vec![]);
        });

        let dump = ssa.to_string(&DefaultSSAAnnotator);
        assert!(dump.contains("memcpy"));
        assert!(dump.contains("count="));
    }
}
