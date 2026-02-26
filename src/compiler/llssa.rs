use crate::compiler::ir::r#type::SSAType;
use crate::compiler::ssa::{
    Block, BlockId, Function, FunctionId, Instruction, SSA, Terminator, ValueId,
};
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
}

impl Display for IntCmpOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntCmpOp::Eq => "eq",
            IntCmpOp::ULt => "ult",
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
    InsertField {
        result: ValueId,
        base: ValueId,
        struct_type: LLStruct,
        field: usize,
        value: ValueId,
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

    // ── VM / Constraint ─────────────────────────────────────────────────
    /// R1CS constraint: a * b = c
    Constrain {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    /// Write a field element to the witness tape
    WriteWitness {
        value: ValueId,
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
            LLOp::InsertField {
                base, value: val, ..
            } => vec![base, val].into_iter(),

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

            // VM / Constraint
            LLOp::Constrain { a, b, c } => vec![a, b, c].into_iter(),
            LLOp::WriteWitness { value } => vec![value].into_iter(),
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
            | LLOp::InsertField { result, .. }
            | LLOp::HeapAlloc { result, .. }
            | LLOp::Load { result, .. }
            | LLOp::StructFieldPtr { result, .. }
            | LLOp::ArrayElemPtr { result, .. }
            | LLOp::Select { result, .. }
            | LLOp::GlobalAddr { result, .. } => vec![result].into_iter(),

            // Multi-result
            LLOp::Call { results, .. } => results.iter().collect::<Vec<_>>().into_iter(),

            // No result
            LLOp::Free { .. }
            | LLOp::Store { .. }
            | LLOp::Memcpy { .. }
            | LLOp::Constrain { .. }
            | LLOp::WriteWitness { .. }
            | LLOp::Trap => vec![].into_iter(),
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
            LLOp::InsertField {
                base, value: val, ..
            } => vec![base, val].into_iter(),

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

            // VM / Constraint
            LLOp::Constrain { a, b, c } => vec![a, b, c].into_iter(),
            LLOp::WriteWitness { value } => vec![value].into_iter(),
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
            LLOp::InsertField {
                result,
                base,
                value,
                ..
            } => vec![result, base, value].into_iter(),

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

            // VM / Constraint
            LLOp::Constrain { a, b, c } => vec![a, b, c].into_iter(),
            LLOp::WriteWitness { value } => vec![value].into_iter(),
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
            LLOp::InsertField {
                result,
                base,
                struct_type,
                field,
                value,
            } => {
                format!(
                    "{} = insert_field {}, {}, {}, {}",
                    v(*result),
                    vr(*base),
                    struct_type,
                    field,
                    vr(*value)
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
            LLOp::Constrain { a, b, c } => {
                format!("constrain {}, {}, {}", vr(*a), vr(*b), vr(*c))
            }
            LLOp::WriteWitness { value } => {
                format!("write_witness {}", vr(*value))
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

// ═══════════════════════════════════════════════════════════════════════════════
// Builder methods
// ═══════════════════════════════════════════════════════════════════════════════

impl LLFunction {
    // ── Constants ────────────────────────────────────────────────────────

    pub fn push_int_const(&mut self, block_id: BlockId, bits: u32, value: u64) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::IntConst {
                result,
                bits,
                value,
            });
        result
    }

    pub fn push_null_ptr(&mut self, block_id: BlockId) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::NullPtr { result });
        result
    }

    // ── Integer Arithmetic ──────────────────────────────────────────────

    pub fn push_int_arith(
        &mut self,
        block_id: BlockId,
        kind: IntArithOp,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::IntArith { kind, result, a, b });
        result
    }

    pub fn push_int_add(&mut self, block_id: BlockId, a: ValueId, b: ValueId) -> ValueId {
        self.push_int_arith(block_id, IntArithOp::Add, a, b)
    }

    pub fn push_int_sub(&mut self, block_id: BlockId, a: ValueId, b: ValueId) -> ValueId {
        self.push_int_arith(block_id, IntArithOp::Sub, a, b)
    }

    pub fn push_int_mul(&mut self, block_id: BlockId, a: ValueId, b: ValueId) -> ValueId {
        self.push_int_arith(block_id, IntArithOp::Mul, a, b)
    }

    pub fn push_not(&mut self, block_id: BlockId, value: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::Not { result, value });
        result
    }

    // ── Integer Comparison ──────────────────────────────────────────────

    pub fn push_int_cmp(
        &mut self,
        block_id: BlockId,
        kind: IntCmpOp,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::IntCmp { kind, result, a, b });
        result
    }

    pub fn push_int_eq(&mut self, block_id: BlockId, a: ValueId, b: ValueId) -> ValueId {
        self.push_int_cmp(block_id, IntCmpOp::Eq, a, b)
    }

    pub fn push_int_ult(&mut self, block_id: BlockId, a: ValueId, b: ValueId) -> ValueId {
        self.push_int_cmp(block_id, IntCmpOp::ULt, a, b)
    }

    // ── Width conversion ────────────────────────────────────────────────

    pub fn push_truncate(&mut self, block_id: BlockId, value: ValueId, to_bits: u32) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::Truncate {
                result,
                value,
                to_bits,
            });
        result
    }

    pub fn push_zext(&mut self, block_id: BlockId, value: ValueId, to_bits: u32) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id).push_instruction(LLOp::ZExt {
            result,
            value,
            to_bits,
        });
        result
    }

    // ── Field Arithmetic ────────────────────────────────────────────────

    pub fn push_field_arith(
        &mut self,
        block_id: BlockId,
        kind: FieldArithOp,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::FieldArith { kind, result, a, b });
        result
    }

    pub fn push_field_neg(&mut self, block_id: BlockId, src: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::FieldNeg { result, src });
        result
    }

    pub fn push_field_eq(&mut self, block_id: BlockId, a: ValueId, b: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::FieldEq { result, a, b });
        result
    }

    pub fn push_field_to_limbs(&mut self, block_id: BlockId, src: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::FieldToLimbs { result, src });
        result
    }

    pub fn push_field_from_limbs(&mut self, block_id: BlockId, limbs: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::FieldFromLimbs { result, limbs });
        result
    }

    // ── Aggregate ───────────────────────────────────────────────────────

    pub fn push_mk_struct(
        &mut self,
        block_id: BlockId,
        struct_type: LLStruct,
        fields: Vec<ValueId>,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::MkStruct {
                result,
                struct_type,
                fields,
            });
        result
    }

    pub fn push_extract_field(
        &mut self,
        block_id: BlockId,
        value: ValueId,
        struct_type: LLStruct,
        field: usize,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::ExtractField {
                result,
                value,
                struct_type,
                field,
            });
        result
    }

    pub fn push_insert_field(
        &mut self,
        block_id: BlockId,
        base: ValueId,
        struct_type: LLStruct,
        field: usize,
        value: ValueId,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::InsertField {
                result,
                base,
                struct_type,
                field,
                value,
            });
        result
    }

    // ── Memory ──────────────────────────────────────────────────────────

    pub fn push_heap_alloc(
        &mut self,
        block_id: BlockId,
        struct_type: LLStruct,
        flex_count: Option<ValueId>,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::HeapAlloc {
                result,
                struct_type,
                flex_count,
            });
        result
    }

    pub fn push_free(&mut self, block_id: BlockId, ptr: ValueId) {
        self.get_block_mut(block_id)
            .push_instruction(LLOp::Free { ptr });
    }

    pub fn push_load(&mut self, block_id: BlockId, ptr: ValueId, ty: LLType) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::Load { result, ptr, ty });
        result
    }

    pub fn push_store(&mut self, block_id: BlockId, ptr: ValueId, value: ValueId) {
        self.get_block_mut(block_id)
            .push_instruction(LLOp::Store { ptr, value });
    }

    pub fn push_struct_field_ptr(
        &mut self,
        block_id: BlockId,
        ptr: ValueId,
        struct_type: LLStruct,
        field: usize,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::StructFieldPtr {
                result,
                ptr,
                struct_type,
                field,
            });
        result
    }

    pub fn push_array_elem_ptr(
        &mut self,
        block_id: BlockId,
        ptr: ValueId,
        elem_type: LLStruct,
        index: ValueId,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::ArrayElemPtr {
                result,
                ptr,
                elem_type,
                index,
            });
        result
    }

    pub fn push_memcpy(
        &mut self,
        block_id: BlockId,
        dst: ValueId,
        src: ValueId,
        struct_type: LLStruct,
        count: Option<ValueId>,
    ) {
        self.get_block_mut(block_id).push_instruction(LLOp::Memcpy {
            dst,
            src,
            struct_type,
            count,
        });
    }

    // ── Selection ───────────────────────────────────────────────────────

    pub fn push_select(
        &mut self,
        block_id: BlockId,
        cond: ValueId,
        if_t: ValueId,
        if_f: ValueId,
    ) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id).push_instruction(LLOp::Select {
            result,
            cond,
            if_t,
            if_f,
        });
        result
    }

    // ── Calls ───────────────────────────────────────────────────────────

    pub fn push_call(
        &mut self,
        block_id: BlockId,
        func: FunctionId,
        args: Vec<ValueId>,
        num_results: usize,
    ) -> Vec<ValueId> {
        let results: Vec<ValueId> = (0..num_results).map(|_| self.fresh_value()).collect();
        self.get_block_mut(block_id).push_instruction(LLOp::Call {
            results: results.clone(),
            func,
            args,
        });
        results
    }

    // ── Globals ─────────────────────────────────────────────────────────

    pub fn push_global_addr(&mut self, block_id: BlockId, global_id: usize) -> ValueId {
        let result = self.fresh_value();
        self.get_block_mut(block_id)
            .push_instruction(LLOp::GlobalAddr { result, global_id });
        result
    }

    // ── VM / Constraint ─────────────────────────────────────────────────

    pub fn push_constrain(&mut self, block_id: BlockId, a: ValueId, b: ValueId, c: ValueId) {
        self.get_block_mut(block_id)
            .push_instruction(LLOp::Constrain { a, b, c });
    }

    pub fn push_write_witness(&mut self, block_id: BlockId, value: ValueId) {
        self.get_block_mut(block_id)
            .push_instruction(LLOp::WriteWitness { value });
    }

    // ── Trap ────────────────────────────────────────────────────────────

    pub fn push_trap(&mut self, block_id: BlockId) {
        self.get_block_mut(block_id).push_instruction(LLOp::Trap);
    }

    // ── Terminators ─────────────────────────────────────────────────────

    pub fn terminate_block_with_jmp(
        &mut self,
        block_id: BlockId,
        destination: BlockId,
        arguments: Vec<ValueId>,
    ) {
        self.get_block_mut(block_id)
            .set_terminator(Terminator::Jmp(destination, arguments));
    }

    pub fn terminate_block_with_jmp_if(
        &mut self,
        block_id: BlockId,
        condition: ValueId,
        then_destination: BlockId,
        else_destination: BlockId,
    ) {
        self.get_block_mut(block_id)
            .set_terminator(Terminator::JmpIf(
                condition,
                then_destination,
                else_destination,
            ));
    }

    pub fn terminate_block_with_return(&mut self, block_id: BlockId, return_values: Vec<ValueId>) {
        self.get_block_mut(block_id)
            .set_terminator(Terminator::Return(return_values));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
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
        let main_id = ssa.get_main_id();
        let func = ssa.get_main_mut();
        let entry = func.get_entry_id();

        // Build: x = 42; y = 7; z = x + y; return z
        let x = func.push_int_const(entry, 64, 42);
        let y = func.push_int_const(entry, 64, 7);
        let z = func.push_int_add(entry, x, y);
        func.terminate_block_with_return(entry, vec![z]);

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

        let l0 = func.push_int_const(entry, 64, 1);
        let l1 = func.push_int_const(entry, 64, 0);
        let l2 = func.push_int_const(entry, 64, 0);
        let l3 = func.push_int_const(entry, 64, 0);
        let s = func.push_mk_struct(entry, field_elem.clone(), vec![l0, l1, l2, l3]);
        let f0 = func.push_extract_field(entry, s, field_elem.clone(), 0);
        let new_val = func.push_int_const(entry, 64, 99);
        let s2 = func.push_insert_field(entry, s, field_elem, 0, new_val);
        func.terminate_block_with_return(entry, vec![f0, s2]);

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("mk_struct"));
        assert!(dump.contains("extract_field"));
        assert!(dump.contains("insert_field"));
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

        let arr = func.push_heap_alloc(entry, rc_array.clone(), None);
        let rc_ptr = func.push_struct_field_ptr(entry, arr, rc_array.clone(), 0);
        let rc_word = func.push_struct_field_ptr(entry, rc_ptr, rc_header, 0);
        let one = func.push_int_const(entry, 64, 1);
        func.push_store(entry, rc_word, one);

        let data = func.push_struct_field_ptr(entry, arr, rc_array, 1);
        let idx = func.push_int_const(entry, 64, 0);
        let elem_ptr = func.push_array_elem_ptr(entry, data, field_elem, idx);
        let loaded = func.push_load(entry, elem_ptr, LLType::i64());
        func.push_free(entry, arr);
        func.terminate_block_with_return(entry, vec![loaded]);

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

        let a = func.push_int_const(entry, 64, 1);
        let b = func.push_int_const(entry, 64, 2);
        let results = func.push_call(entry, helper_id, vec![a, b], 1);
        let cond = func.push_int_eq(entry, results[0], a);
        let selected = func.push_select(entry, cond, a, b);
        func.terminate_block_with_return(entry, vec![selected]);

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

        let l0 = func.push_int_const(entry, 64, 1);
        let l1 = func.push_int_const(entry, 64, 0);
        let l2 = func.push_int_const(entry, 64, 0);
        let l3 = func.push_int_const(entry, 64, 0);
        let a = func.push_mk_struct(entry, field_elem.clone(), vec![l0, l1, l2, l3]);
        let b = func.push_mk_struct(entry, field_elem, vec![l0, l1, l2, l3]);

        let c = func.push_field_arith(entry, FieldArithOp::Add, a, b);
        let d = func.push_field_neg(entry, c);
        let eq = func.push_field_eq(entry, c, d);
        let limbs = func.push_field_to_limbs(entry, d);
        let back = func.push_field_from_limbs(entry, limbs);
        func.terminate_block_with_return(entry, vec![eq, back]);

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

        let x = func.push_int_const(entry, 64, 256);
        let narrow = func.push_truncate(entry, x, 8);
        let wide = func.push_zext(entry, narrow, 64);
        let gp = func.push_global_addr(entry, 3);
        func.push_store(entry, gp, wide);
        func.push_trap(entry);

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

        let x = func.push_int_const(entry, 64, 42);
        let zero = func.push_int_const(entry, 64, 0);
        let cond = func.push_int_eq(entry, x, zero);
        func.terminate_block_with_jmp_if(entry, cond, then_blk, else_blk);

        let one = func.push_int_const(then_blk, 64, 1);
        func.terminate_block_with_jmp(then_blk, merge_blk, vec![one]);

        let two = func.push_int_const(else_blk, 64, 2);
        func.terminate_block_with_jmp(else_blk, merge_blk, vec![two]);

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
        let dst = func.push_null_ptr(entry);
        let src = func.push_null_ptr(entry);
        let count = func.push_int_const(entry, 64, 10);
        func.push_memcpy(entry, dst, src, elem, Some(count));
        func.terminate_block_with_return(entry, vec![]);

        let dump = ssa.to_string(&DefaultSsaAnnotator);
        assert!(dump.contains("memcpy"));
        assert!(dump.contains("count="));
    }
}
