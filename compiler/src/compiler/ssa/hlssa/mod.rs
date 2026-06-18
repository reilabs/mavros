//! The high-level SSA representation used in the compiler and its associated types.

pub mod builder;
pub mod type_system;

use itertools::Itertools;
use std::fmt::Display;

use crate::compiler::ssa::{
    Block, Function, FunctionId, Instruction, SSA, SSAConstants, SSAConstantsSnapshot, ValueId,
};
pub use type_system::{MAX_SUPPORTED_SIGNED_BITS, MAX_SUPPORTED_UNSIGNED_BITS, Type, TypeExpr};

// HLSSA
// ================================================================================================

/// The high-level SSA is designed for domain-level analysis without concretizing runtime details.
pub type HLSSA = SSA<OpCode, Type, Constant>;

impl HLSSA {
    pub fn new() -> Self {
        Self::with_main("main".to_string())
    }
}

// CONSTANT STORAGE
// ================================================================================================

/// Concrete constants side-table type for the high-level SSA.
pub type HLSSAConstants = SSAConstants<Constant>;

/// Owned, lock-free snapshot of the high-level SSA's constants, as handed out by
/// [`SSA::const_snapshot`](crate::compiler::ssa::SSA::const_snapshot).
pub type HLSSAConstantsSnapshot = SSAConstantsSnapshot<Constant>;

// HLSSA OPCODES
// ================================================================================================

/// The high-level opcodes for use with the high-level SSA.
#[derive(Debug, Clone)]
pub enum OpCode {
    Cmp {
        kind: CmpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    },
    BinaryArithOp {
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    },
    Cast {
        result: ValueId,
        value: ValueId,
        target: CastTarget,
    },
    SExt {
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    },
    BitRange {
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    },
    Not {
        result: ValueId,
        value: ValueId,
    },
    MkSeq {
        result: ValueId,
        elems: Vec<ValueId>,
        seq_type: SequenceTargetType,
        elem_type: Type,
    },
    MkSeqOfBlob {
        result: ValueId,
        element_type: Type,
        blob: ValueId,
    },
    MkRepeated {
        result: ValueId,
        element: ValueId,
        seq_type: SequenceTargetType,
        count: usize,
        elem_type: Type,
    },
    Alloc {
        result: ValueId,
        value: ValueId,
    },
    Store {
        ptr: ValueId,
        value: ValueId,
    },
    Load {
        result: ValueId,
        ptr: ValueId,
    },
    Assert {
        value: ValueId,
    },
    AssertCmp {
        kind: CmpKind,
        lhs: ValueId,
        rhs: ValueId,
    },
    AssertR1C {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    Call {
        results: Vec<ValueId>,
        function: CallTarget,
        args: Vec<ValueId>,
        unconstrained: bool,
    },
    ArrayGet {
        result: ValueId,
        array: ValueId,
        index: ValueId,
    },
    ArraySet {
        result: ValueId,
        array: ValueId,
        index: ValueId,
        value: ValueId,
    },
    SlicePush {
        dir: SliceOpDir,
        result: ValueId,
        slice: ValueId,
        values: Vec<ValueId>,
    },
    SliceLen {
        result: ValueId,
        slice: ValueId,
    },
    Select {
        result: ValueId,
        cond: ValueId,
        if_t: ValueId,
        if_f: ValueId,
    },
    ToBits {
        result: ValueId,
        value: ValueId,
        endianness: Endianness,
        count: usize,
    },
    ToRadix {
        result: ValueId,
        value: ValueId,
        radix: Radix<ValueId>,
        endianness: Endianness,
        count: usize,
    },
    MemOp {
        kind: RefCountOp,
        value: ValueId,
    },
    WriteWitness {
        result: Option<ValueId>,
        value: ValueId,
        pinned: bool,
    },
    FreshWitness {
        result: ValueId,
        result_type: Type,
    },
    NextDCoeff {
        result: ValueId,
    },
    BumpD {
        matrix: DMatrix,
        variable: ValueId,
        sensitivity: ValueId,
    },
    Constrain {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    Lookup {
        target: LookupTarget<ValueId>,
        args: Vec<ValueId>,
        flag: ValueId,
    },
    DLookup {
        target: LookupTarget<ValueId>,
        args: Vec<ValueId>,
        flag: ValueId,
    },
    MulConst {
        result: ValueId,
        const_val: ValueId,
        var: ValueId,
    },
    Rangecheck {
        value: ValueId,
        max_bits: usize,
    },
    ReadGlobal {
        result: ValueId,
        offset: u64,
        result_type: Type,
    },
    TupleProj {
        result: ValueId,
        tuple: ValueId,
        idx: usize,
    },
    TupleRefProj {
        result: ValueId,
        tuple_ref: ValueId,
        idx: usize,
    },
    MkTuple {
        result: ValueId,
        elems: Vec<ValueId>,
        element_types: Vec<Type>,
    },
    Todo {
        payload: String,
        results: Vec<ValueId>,
        result_types: Vec<Type>,
    },
    InitGlobal {
        global: usize,
        value: ValueId,
    },
    DropGlobal {
        global: usize,
    },
    Spread {
        result: ValueId,
        value: ValueId,
        /// Number of input bits (1..=16).
        bits: u8,
    },
    Unspread {
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
        /// Number of input bits per half (1..=16).
        bits: u8,
    },
    Guard {
        condition: ValueId,
        inner: Box<OpCode>,
    },
}

impl Instruction for OpCode {
    fn get_static_call_targets(&self) -> Vec<FunctionId> {
        match self {
            OpCode::Call {
                function: CallTarget::Static(id),
                ..
            } => vec![*id],
            OpCode::Guard { inner, .. } => inner.get_static_call_targets(),
            _ => vec![],
        }
    }

    fn map_call_targets(&mut self, f: &mut dyn FnMut(FunctionId) -> FunctionId) {
        match self {
            OpCode::Call {
                function: CallTarget::Static(id),
                ..
            } => *id = f(*id),
            OpCode::Guard { inner, .. } => inner.map_call_targets(f),
            _ => {}
        }
    }

    fn display_instruction(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        annotate_value: &dyn Fn(ValueId) -> String,
    ) -> String {
        match self {
            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let op_str = match kind {
                    CmpKind::Lt => "<",
                    CmpKind::Eq => "==",
                };
                format!(
                    "v{}{} = v{} {} v{}",
                    result.0,
                    annotate_value(*result),
                    lhs.0,
                    op_str,
                    rhs.0
                )
            }
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let op_str = match kind {
                    BinaryArithOpKind::Add => "+",
                    BinaryArithOpKind::Sub => "-",
                    BinaryArithOpKind::Mul => "*",
                    BinaryArithOpKind::Div => "/",
                    BinaryArithOpKind::And => "&",
                    BinaryArithOpKind::Or => "|",
                    BinaryArithOpKind::Xor => "^",
                    BinaryArithOpKind::Shl => "<<",
                    BinaryArithOpKind::Shr => ">>",
                    BinaryArithOpKind::Mod => "%",
                };
                format!(
                    "v{}{} = v{} {} v{}",
                    result.0,
                    annotate_value(*result),
                    lhs.0,
                    op_str,
                    rhs.0
                )
            }
            OpCode::Alloc { result, value } => format!(
                "v{}{} = alloc(v{})",
                result.0,
                annotate_value(*result),
                value.0
            ),
            OpCode::Store { ptr, value } => {
                format!("*v{}{} = v{}", ptr.0, annotate_value(*ptr), value.0)
            }
            OpCode::Load { result, ptr } => {
                format!("v{}{} = *v{}", result.0, annotate_value(*result), ptr.0)
            }
            OpCode::Assert { value } => format!("assert v{}", value.0),
            OpCode::AssertCmp { kind, lhs, rhs } => {
                let op_str = match kind {
                    CmpKind::Lt => "<",
                    CmpKind::Eq => "==",
                };
                format!("assert v{} {} v{}", lhs.0, op_str, rhs.0)
            }
            OpCode::AssertR1C {
                a: lhs,
                b: rhs,
                c: cond,
            } => {
                format!("assert v{} * v{} - v{} == 0", lhs.0, rhs.0, cond.0)
            }
            OpCode::Call {
                results: result,
                function,
                args,
                unconstrained,
            } => {
                let args_str = args.iter().map(|v| format!("v{}", v.0)).join(", ");
                let result_str = result
                    .iter()
                    .map(|v| format!("v{}{}", v.0, annotate_value(*v)))
                    .join(", ");
                let call_prefix = if *unconstrained {
                    "call_unconstrained"
                } else {
                    "call"
                };
                match function {
                    CallTarget::Static(fn_id) => {
                        format!(
                            "{} = {} {}@{}({})",
                            result_str,
                            call_prefix,
                            func_name(*fn_id),
                            fn_id.0,
                            args_str
                        )
                    }
                    CallTarget::Dynamic(fn_ptr) => {
                        format!(
                            "{} = {}_indirect v{}({})",
                            result_str, call_prefix, fn_ptr.0, args_str
                        )
                    }
                }
            }
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                format!(
                    "v{}{} = v{}[v{}]",
                    result.0,
                    annotate_value(*result),
                    array.0,
                    index.0
                )
            }
            OpCode::ArraySet {
                result,
                array,
                index,
                value: element,
            } => {
                format!(
                    "v{}{} = (v{}[v{}] = v{})",
                    result.0,
                    annotate_value(*result),
                    array.0,
                    index.0,
                    element.0
                )
            }
            OpCode::SlicePush {
                dir,
                result,
                slice,
                values,
            } => {
                let dir_str = match dir {
                    SliceOpDir::Front => "front",
                    SliceOpDir::Back => "back",
                };
                let values_str = values.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!(
                    "v{}{} = slice_push_{}(v{}, [{}])",
                    result.0,
                    annotate_value(*result),
                    dir_str,
                    slice.0,
                    values_str
                )
            }
            OpCode::SliceLen { result, slice } => {
                format!(
                    "v{}{} = slice_len(v{})",
                    result.0,
                    annotate_value(*result),
                    slice.0
                )
            }
            OpCode::Select {
                result,
                cond,
                if_t: then,
                if_f: otherwise,
            } => {
                format!(
                    "v{}{} = v{} ? v{} : v{}",
                    result.0,
                    annotate_value(*result),
                    cond.0,
                    then.0,
                    otherwise.0
                )
            }
            OpCode::WriteWitness {
                result,
                value,
                pinned,
            } => {
                let r_str = if let Some(result) = result {
                    format!("v{}{} = ", result.0, annotate_value(*result))
                } else {
                    "".to_string()
                };
                let pinned_str = if *pinned { " [pinned]" } else { "" };
                format!("{}write_witness(v{}){}", r_str, value.0, pinned_str)
            }
            OpCode::FreshWitness {
                result,
                result_type: typ,
            } => {
                format!(
                    "v{}{} = fresh_witness(): {}",
                    result.0,
                    annotate_value(*result),
                    typ
                )
            }
            OpCode::Constrain { a, b, c } => {
                format!("constrain_r1c(v{} * v{} - v{} == 0)", a.0, b.0, c.0)
            }
            OpCode::Lookup { target, args, flag } => {
                let args_str = args.iter().map(|v| format!("v{}", v.0)).join(", ");
                let target_str = match target {
                    LookupTarget::Rangecheck(n) => format!("rngchk({})", n),
                    LookupTarget::DynRangecheck(v) => format!("rngchk(_ < v{})", v.0),
                    LookupTarget::Array(arr) => format!("v{}", arr.0),
                    LookupTarget::Spread(n) => format!("spread({})", n),
                };
                format!(
                    "constrain_lookup({}, ({}), flag=v{})",
                    target_str, args_str, flag.0
                )
            }
            OpCode::NextDCoeff { result } => {
                format!("v{}{} = next_d_coeff()", result.0, annotate_value(*result))
            }
            OpCode::BumpD {
                matrix,
                variable: result,
                sensitivity: value,
            } => {
                let matrix_str = match matrix {
                    DMatrix::A => "A",
                    DMatrix::B => "B",
                    DMatrix::C => "C",
                };
                format!("∂{} / ∂v{} += v{}", matrix_str, result.0, value.0)
            }
            OpCode::DLookup { target, args, flag } => {
                let args_str = args.iter().map(|v| format!("v{}", v.0)).join(", ");
                let target_str = match target {
                    LookupTarget::Rangecheck(n) => format!("rngchk({})", n),
                    LookupTarget::DynRangecheck(v) => format!("rngchk(_ < v{})", v.0),
                    LookupTarget::Array(arr) => format!("v{}", arr.0),
                    LookupTarget::Spread(n) => format!("spread({})", n),
                };
                format!("∂lookup({}, ({}), flag=v{})", target_str, args_str, flag.0)
            }
            OpCode::MkSeq {
                result,
                elems: values,
                seq_type,
                elem_type: typ,
            } => {
                let values_str = values.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!(
                    "v{}{} = [{}] : {} of {}",
                    result.0,
                    annotate_value(*result),
                    values_str,
                    seq_type,
                    typ
                )
            }
            OpCode::MkSeqOfBlob {
                result,
                element_type: typ,
                blob,
            } => {
                format!(
                    "v{}{} = mk_seq_of_blob(v{}) of {}",
                    result.0,
                    annotate_value(*result),
                    blob.0,
                    typ
                )
            }
            OpCode::MkRepeated {
                result,
                element,
                seq_type,
                count,
                elem_type,
            } => {
                format!(
                    "v{}{} = [v{}; {}] : {} of {}",
                    result.0,
                    annotate_value(*result),
                    element.0,
                    count,
                    seq_type,
                    elem_type
                )
            }
            OpCode::Cast {
                result,
                value,
                target,
            } => {
                format!(
                    "v{}{} = cast v{} to {}",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    target
                )
            }
            OpCode::SExt {
                result,
                value,
                from_bits: in_bits,
                to_bits: out_bits,
            } => {
                format!(
                    "v{}{} = sext v{} from {} bits to {} bits",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    in_bits,
                    out_bits
                )
            }
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => {
                format!(
                    "v{}{} = bit_range(v{}, {}, {})",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    offset,
                    width
                )
            }
            OpCode::Not { result, value } => {
                format!("v{}{} = ~v{}", result.0, annotate_value(*result), value.0)
            }
            OpCode::ToBits {
                result,
                value,
                endianness,
                count: output_size,
            } => {
                format!(
                    "v{}{} = to_bits v{} (endianness: {}, size: {})",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    endianness,
                    output_size
                )
            }
            OpCode::ToRadix {
                result,
                value,
                radix,
                endianness,
                count: output_size,
            } => {
                let radix_str = match radix {
                    Radix::Bytes => "bytes".to_string(),
                    Radix::Dyn(radix) => format!("v{}", radix.0),
                };
                format!(
                    "v{}{} = to_radix v{} {} (endianness: {}, size: {})",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    radix_str,
                    endianness,
                    output_size
                )
            }
            OpCode::MemOp { kind, value } => {
                let name = match kind {
                    RefCountOp::Bump(n) => format!("inc_rc[+{}]", n),
                    RefCountOp::Drop => "drop".to_string(),
                };
                format!("{}(v{})", name, value.0)
            }
            OpCode::MulConst {
                result,
                const_val: constant,
                var,
            } => {
                format!(
                    "v{}{} = mul_const(v{}, v{})",
                    result.0,
                    annotate_value(*result),
                    constant.0,
                    var.0
                )
            }
            OpCode::Rangecheck {
                value: val,
                max_bits,
            } => {
                format!("rangecheck(v{}, {})", val.0, max_bits)
            }
            OpCode::ReadGlobal {
                result,
                offset: index,
                result_type: typ,
            } => {
                format!(
                    "v{}{} = read_global(g{}, {})",
                    result.0,
                    annotate_value(*result),
                    index,
                    typ
                )
            }
            OpCode::TupleProj { result, tuple, idx } => {
                format!(
                    "v{}{} = v{}.{}",
                    result.0,
                    annotate_value(*result),
                    tuple.0,
                    idx
                )
            }
            OpCode::TupleRefProj {
                result,
                tuple_ref,
                idx,
            } => {
                format!(
                    "v{}{} = ref_proj(v{}, {})",
                    result.0,
                    annotate_value(*result),
                    tuple_ref.0,
                    idx
                )
            }
            OpCode::MkTuple {
                result,
                elems,
                element_types: _,
            } => {
                let elems_str = elems.iter().map(|v| format!("v{}", v.0)).join(", ");
                format!("v{}{} = ({})", result.0, annotate_value(*result), elems_str)
            }
            OpCode::Todo {
                payload,
                results,
                result_types,
            } => {
                let results_str = results
                    .iter()
                    .zip(result_types.iter())
                    .map(|(r, tp)| format!("v{}: {}", r.0, tp))
                    .join(", ");
                format!("todo(\"{}\", [{}])", payload, results_str)
            }
            OpCode::InitGlobal { global, value } => {
                format!("init_global({}, v{})", global, value.0)
            }
            OpCode::DropGlobal { global } => {
                format!("drop_global({})", global)
            }
            OpCode::Spread {
                result,
                value,
                bits,
            } => {
                format!(
                    "v{}{} = spread(v{}, {bits})",
                    result.0,
                    annotate_value(*result),
                    value.0,
                )
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                bits,
            } => {
                format!(
                    "v{}{}, v{}{} = unspread(v{}, {bits})",
                    result_odd.0,
                    annotate_value(*result_odd),
                    result_even.0,
                    annotate_value(*result_even),
                    value.0,
                )
            }
            OpCode::Guard { condition, inner } => {
                format!(
                    "guard(v{}) {{ {} }}",
                    condition.0,
                    inner.display_instruction(func_name, annotate_value)
                )
            }
        }
    }

    fn get_inputs(&self) -> impl Iterator<Item = &ValueId> {
        match self {
            Self::Alloc {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::FreshWitness {
                result: _,
                result_type: _,
            }
            | Self::NextDCoeff { result: _ } => vec![].into_iter(),
            Self::Cmp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::BinaryArithOp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::ArrayGet {
                result: _,
                array: b,
                index: c,
            } => vec![b, c].into_iter(),
            Self::Spread {
                result: _,
                value: v,
                ..
            } => vec![v].into_iter(),
            Self::Unspread {
                result_odd: _,
                result_even: _,
                value: v,
                ..
            } => vec![v].into_iter(),
            Self::ArraySet {
                result: _,
                array: b,
                index: c,
                value: d,
            } => vec![b, c, d].into_iter(),
            Self::SlicePush {
                dir: _,
                result: _,
                slice: b,
                values: c,
            } => {
                let mut ret_vec = vec![b];
                ret_vec.extend(c.iter());
                ret_vec.into_iter()
            }
            Self::SliceLen {
                result: _,
                slice: b,
            } => vec![b].into_iter(),
            Self::AssertCmp {
                kind: _,
                lhs: b,
                rhs: c,
            }
            | Self::Store { ptr: b, value: c }
            | Self::BumpD {
                matrix: _,
                variable: b,
                sensitivity: c,
            }
            | Self::MulConst {
                result: _,
                const_val: b,
                var: c,
            } => vec![b, c].into_iter(),
            Self::Assert { value: c }
            | Self::Load { result: _, ptr: c }
            | Self::WriteWitness {
                result: _,
                value: c,
                pinned: _,
            }
            | Self::Cast {
                result: _,
                value: c,
                target: _,
            }
            | Self::SExt {
                result: _,
                value: c,
                from_bits: _,
                to_bits: _,
            }
            | Self::BitRange {
                result: _,
                value: c,
                offset: _,
                width: _,
            } => vec![c].into_iter(),
            Self::Call {
                results: _,
                function,
                args: a,
                unconstrained: _,
            } => {
                let mut ret_vec = Vec::new();
                if let CallTarget::Dynamic(fn_ptr) = function {
                    ret_vec.push(fn_ptr);
                }
                ret_vec.extend(a.iter());
                ret_vec.into_iter()
            }
            Self::MkSeq {
                result: _,
                elems: inputs,
                seq_type: _,
                elem_type: _,
            } => inputs.iter().collect::<Vec<_>>().into_iter(),
            Self::MkSeqOfBlob { blob, .. } => vec![blob].into_iter(),
            Self::MkRepeated {
                result: _,
                element,
                seq_type: _,
                count: _,
                elem_type: _,
            } => vec![element].into_iter(),
            Self::Select {
                result: _,
                cond: b,
                if_t: c,
                if_f: d,
            }
            | Self::AssertR1C { a: b, b: c, c: d }
            | Self::Constrain { a: b, b: c, c: d } => vec![b, c, d].into_iter(),
            Self::Not {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::ToBits {
                result: _,
                value: v,
                endianness: _,
                count: _,
            } => vec![v].into_iter(),
            Self::ToRadix {
                result: _,
                value: v,
                radix,
                endianness: _,
                count: _,
            } => {
                let mut ret_vec = vec![v];
                match radix {
                    Radix::Bytes => {}
                    Radix::Dyn(radix) => {
                        ret_vec.push(radix);
                    }
                }
                ret_vec.into_iter()
            }
            Self::MemOp { kind: _, value: v } => vec![v].into_iter(),
            Self::Rangecheck {
                value: val,
                max_bits: _,
            } => vec![val].into_iter(),
            Self::ReadGlobal {
                result: _,
                offset: _,
                result_type: _,
            } => vec![].into_iter(),
            Self::Lookup { target, args, flag } | Self::DLookup { target, args, flag } => {
                let mut ret_vec = vec![];
                match target {
                    LookupTarget::Rangecheck(_) | LookupTarget::Spread(_) => {}
                    LookupTarget::DynRangecheck(v) => {
                        ret_vec.push(v);
                    }
                    LookupTarget::Array(arr) => {
                        ret_vec.push(arr);
                    }
                }
                ret_vec.extend(args);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::TupleProj {
                result: _,
                tuple,
                idx: _,
            }
            | Self::TupleRefProj {
                result: _,
                tuple_ref: tuple,
                idx: _,
            } => vec![tuple].into_iter(),
            OpCode::MkTuple {
                result: _,
                elems: e,
                element_types: _,
            } => e.iter().collect::<Vec<_>>().into_iter(),
            Self::Todo { .. } => vec![].into_iter(),
            Self::InitGlobal {
                global: _,
                value: v,
            } => vec![v].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { condition, inner } => {
                let mut ret_vec = vec![condition];
                ret_vec.extend(inner.get_inputs());
                ret_vec.into_iter()
            }
        }
    }

    fn get_results(&self) -> impl Iterator<Item = &ValueId> {
        match self {
            Self::Alloc { result: r, .. }
            | Self::FreshWitness { result: r, .. }
            | Self::Cmp { result: r, .. }
            | Self::BinaryArithOp { result: r, .. }
            | Self::ArrayGet { result: r, .. }
            | Self::ArraySet { result: r, .. }
            | Self::SlicePush { result: r, .. }
            | Self::SliceLen { result: r, .. }
            | Self::Load { result: r, ptr: _ }
            | Self::MkSeq { result: r, .. }
            | Self::MkSeqOfBlob { result: r, .. }
            | Self::MkRepeated { result: r, .. }
            | Self::Select { result: r, .. }
            | Self::Cast { result: r, .. }
            | Self::SExt { result: r, .. }
            | Self::BitRange { result: r, .. }
            | Self::MulConst { result: r, .. }
            | Self::NextDCoeff { result: r }
            | Self::TupleProj { result: r, .. }
            | Self::TupleRefProj { result: r, .. }
            | Self::MkTuple { result: r, .. } => vec![r].into_iter(),
            Self::WriteWitness { result: r, .. } => {
                let ret_vec = r.iter().collect::<Vec<_>>();
                ret_vec.into_iter()
            }
            Self::Call { results: r, .. } => r.iter().collect::<Vec<_>>().into_iter(),
            Self::Constrain { .. }
            | Self::BumpD { .. }
            | Self::MemOp { .. }
            | Self::Store { .. }
            | Self::Assert { .. }
            | Self::AssertCmp { .. }
            | Self::AssertR1C { a: _, b: _, c: _ }
            | Self::Rangecheck { .. } => vec![].into_iter(),
            Self::Not { result: r, .. } | Self::Spread { result: r, .. } => vec![r].into_iter(),
            Self::Unspread {
                result_odd,
                result_even,
                ..
            } => vec![result_odd, result_even].into_iter(),
            Self::ToBits { result: r, .. } => vec![r].into_iter(),
            Self::ToRadix { result: r, .. } => vec![r].into_iter(),
            Self::ReadGlobal { result: r, .. } => vec![r].into_iter(),
            Self::Lookup { .. } | Self::DLookup { .. } => vec![].into_iter(),
            Self::Todo { results, .. } => {
                let ret_vec: Vec<&ValueId> = results.iter().collect();
                ret_vec.into_iter()
            }
            Self::InitGlobal { .. } => vec![].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { inner, .. } => inner.get_results().collect::<Vec<_>>().into_iter(),
        }
    }

    fn get_results_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            Self::Alloc { result: r, .. }
            | Self::FreshWitness { result: r, .. }
            | Self::Cmp { result: r, .. }
            | Self::BinaryArithOp { result: r, .. }
            | Self::ArrayGet { result: r, .. }
            | Self::ArraySet { result: r, .. }
            | Self::SlicePush { result: r, .. }
            | Self::SliceLen { result: r, .. }
            | Self::Load { result: r, ptr: _ }
            | Self::MkSeq { result: r, .. }
            | Self::MkSeqOfBlob { result: r, .. }
            | Self::MkRepeated { result: r, .. }
            | Self::Select { result: r, .. }
            | Self::Cast { result: r, .. }
            | Self::SExt { result: r, .. }
            | Self::BitRange { result: r, .. }
            | Self::MulConst { result: r, .. }
            | Self::NextDCoeff { result: r }
            | Self::TupleProj { result: r, .. }
            | Self::TupleRefProj { result: r, .. }
            | Self::MkTuple { result: r, .. } => vec![r].into_iter(),
            Self::WriteWitness { result: r, .. } => {
                let ret_vec = r.iter_mut().collect::<Vec<_>>();
                ret_vec.into_iter()
            }
            Self::Call { results: r, .. } => r.iter_mut().collect::<Vec<_>>().into_iter(),
            Self::Constrain { .. }
            | Self::BumpD { .. }
            | Self::MemOp { .. }
            | Self::Store { .. }
            | Self::Assert { .. }
            | Self::AssertCmp { .. }
            | Self::AssertR1C { a: _, b: _, c: _ }
            | Self::Rangecheck { .. } => vec![].into_iter(),
            Self::Not { result: r, .. } | Self::Spread { result: r, .. } => vec![r].into_iter(),
            Self::Unspread {
                result_odd,
                result_even,
                ..
            } => vec![result_odd, result_even].into_iter(),
            Self::ToBits { result: r, .. } => vec![r].into_iter(),
            Self::ToRadix { result: r, .. } => vec![r].into_iter(),
            Self::ReadGlobal { result: r, .. } => vec![r].into_iter(),
            Self::Lookup { .. } | Self::DLookup { .. } => vec![].into_iter(),
            Self::Todo { results, .. } => {
                let ret_vec: Vec<&mut ValueId> = results.iter_mut().collect();
                ret_vec.into_iter()
            }
            Self::InitGlobal { .. } => vec![].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { inner, .. } => inner.get_results_mut().collect::<Vec<_>>().into_iter(),
        }
    }

    fn get_inputs_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            Self::Alloc {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::FreshWitness {
                result: _,
                result_type: _,
            }
            | Self::NextDCoeff { result: _ } => vec![].into_iter(),
            Self::Cmp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::BinaryArithOp {
                kind: _,
                result: _,
                lhs: b,
                rhs: c,
            }
            | Self::ArrayGet {
                result: _,
                array: b,
                index: c,
            }
            | Self::MulConst {
                result: _,
                const_val: b,
                var: c,
            } => vec![b, c].into_iter(),
            Self::Spread {
                result: _,
                value: v,
                ..
            } => vec![v].into_iter(),
            Self::Unspread {
                result_odd: _,
                result_even: _,
                value: v,
                ..
            } => vec![v].into_iter(),
            Self::ArraySet {
                result: _,
                array: b,
                index: c,
                value: d,
            } => vec![b, c, d].into_iter(),
            Self::SlicePush {
                dir: _,
                result: _,
                slice: b,
                values: c,
            } => {
                let mut ret_vec = vec![b];
                let values_vec: Vec<&mut ValueId> = c.iter_mut().collect();
                ret_vec.extend(values_vec);
                ret_vec.into_iter()
            }
            Self::SliceLen {
                result: _,
                slice: b,
            } => vec![b].into_iter(),
            Self::AssertCmp {
                kind: _,
                lhs: b,
                rhs: c,
            }
            | Self::Store { ptr: b, value: c }
            | Self::BumpD {
                matrix: _,
                variable: b,
                sensitivity: c,
            } => vec![b, c].into_iter(),
            Self::Assert { value: c }
            | Self::Load { result: _, ptr: c }
            | Self::WriteWitness {
                result: _,
                value: c,
                pinned: _,
            }
            | Self::Cast {
                result: _,
                value: c,
                target: _,
            }
            | Self::SExt {
                result: _,
                value: c,
                from_bits: _,
                to_bits: _,
            }
            | Self::BitRange {
                result: _,
                value: c,
                offset: _,
                width: _,
            } => vec![c].into_iter(),
            Self::Call {
                results: _,
                function,
                args: a,
                unconstrained: _,
            } => {
                let mut ret_vec = Vec::new();
                if let CallTarget::Dynamic(fn_ptr) = function {
                    ret_vec.push(fn_ptr);
                }
                ret_vec.extend(a.iter_mut());
                ret_vec.into_iter()
            }
            Self::MkSeq {
                result: _,
                elems: inputs,
                seq_type: _,
                elem_type: _,
            } => inputs.iter_mut().collect::<Vec<_>>().into_iter(),
            Self::MkSeqOfBlob { blob, .. } => vec![blob].into_iter(),
            Self::MkRepeated {
                result: _,
                element,
                seq_type: _,
                count: _,
                elem_type: _,
            } => vec![element].into_iter(),
            Self::MkTuple {
                result: _,
                elems: inputs,
                element_types: _,
            } => inputs.iter_mut().collect::<Vec<_>>().into_iter(),
            Self::Select {
                result: _,
                cond: b,
                if_t: c,
                if_f: d,
            }
            | Self::AssertR1C { a: b, b: c, c: d }
            | Self::Constrain { a: b, b: c, c: d } => vec![b, c, d].into_iter(),
            Self::Not {
                result: _,
                value: v,
            } => vec![v].into_iter(),
            Self::ToBits {
                result: _,
                value: v,
                endianness: _,
                count: _,
            } => vec![v].into_iter(),
            Self::ToRadix {
                result: _,
                value: v,
                radix,
                endianness: _,
                count: _,
            } => {
                let mut ret_vec = vec![v];
                match radix {
                    Radix::Bytes => {}
                    Radix::Dyn(radix) => {
                        ret_vec.push(radix);
                    }
                }
                ret_vec.into_iter()
            }
            Self::MemOp { kind: _, value: v } => vec![v].into_iter(),
            Self::Rangecheck {
                value: val,
                max_bits: _,
            } => vec![val].into_iter(),
            Self::ReadGlobal {
                result: _,
                offset: _,
                result_type: _,
            } => vec![].into_iter(),
            Self::Lookup { target, args, flag } | Self::DLookup { target, args, flag } => {
                let mut ret_vec = vec![];
                match target {
                    LookupTarget::Rangecheck(_) | LookupTarget::Spread(_) => {}
                    LookupTarget::DynRangecheck(v) => {
                        ret_vec.push(v);
                    }
                    LookupTarget::Array(arr) => {
                        ret_vec.push(arr);
                    }
                }
                ret_vec.extend(args);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::TupleProj {
                result: _,
                tuple,
                idx: _,
            }
            | Self::TupleRefProj {
                result: _,
                tuple_ref: tuple,
                idx: _,
            } => vec![tuple].into_iter(),
            Self::Todo { .. } => vec![].into_iter(),
            Self::InitGlobal {
                global: _,
                value: v,
            } => vec![v].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { condition, inner } => {
                let mut ret_vec = vec![condition];
                ret_vec.extend(inner.get_inputs_mut());
                ret_vec.into_iter()
            }
        }
    }

    fn get_operands_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        match self {
            Self::Alloc {
                result: r,
                value: v,
            } => vec![r, v].into_iter(),
            Self::MemOp { kind: _, value: r }
            | Self::FreshWitness {
                result: r,
                result_type: _,
            }
            | Self::NextDCoeff { result: r } => vec![r].into_iter(),
            Self::Cmp {
                kind: _,
                result: a,
                lhs: b,
                rhs: c,
            }
            | Self::BinaryArithOp {
                kind: _,
                result: a,
                lhs: b,
                rhs: c,
            }
            | Self::ArrayGet {
                result: a,
                array: b,
                index: c,
            }
            | Self::MulConst {
                result: a,
                const_val: b,
                var: c,
            } => vec![a, b, c].into_iter(),
            Self::Cast {
                result: a,
                value: b,
                target: _,
            } => vec![a, b].into_iter(),
            Self::SExt {
                result: a,
                value: b,
                from_bits: _,
                to_bits: _,
            }
            | Self::BitRange {
                result: a,
                value: b,
                offset: _,
                width: _,
            } => vec![a, b].into_iter(),
            Self::ArraySet {
                result: a,
                array: b,
                index: c,
                value: d,
            } => vec![a, b, c, d].into_iter(),
            Self::SlicePush {
                dir: _,
                result: a,
                slice: b,
                values: c,
            } => {
                let mut ret_vec = vec![a, b];
                let values_vec = c.iter_mut().collect::<Vec<_>>();
                ret_vec.extend(values_vec);
                ret_vec.into_iter()
            }
            Self::SliceLen {
                result: a,
                slice: b,
            } => vec![a, b].into_iter(),
            Self::AssertR1C { a, b, c } | Self::Constrain { a, b, c } => vec![a, b, c].into_iter(),
            Self::Store { ptr: a, value: b }
            | Self::Load { result: a, ptr: b }
            | Self::AssertCmp {
                kind: _,
                lhs: a,
                rhs: b,
            }
            | Self::BumpD {
                matrix: _,
                variable: a,
                sensitivity: b,
            } => vec![a, b].into_iter(),
            Self::Assert { value: a } => vec![a].into_iter(),
            Self::WriteWitness {
                result: a,
                value: b,
                pinned: _,
            } => {
                let mut ret_vec = a.iter_mut().collect::<Vec<_>>();
                ret_vec.push(b);
                ret_vec.into_iter()
            }
            Self::Call {
                results: r,
                function,
                args: a,
                unconstrained: _,
            } => {
                let mut ret_vec = r.iter_mut().collect::<Vec<_>>();
                if let CallTarget::Dynamic(fn_ptr) = function {
                    ret_vec.push(fn_ptr);
                }
                let args_vec = a.iter_mut().collect::<Vec<_>>();
                ret_vec.extend(args_vec);
                ret_vec.into_iter()
            }
            Self::Lookup { target, args, flag } | Self::DLookup { target, args, flag } => {
                let mut ret_vec = vec![];
                match target {
                    LookupTarget::Rangecheck(_) | LookupTarget::Spread(_) => {}
                    LookupTarget::DynRangecheck(v) => {
                        ret_vec.push(v);
                    }
                    LookupTarget::Array(arr) => {
                        ret_vec.push(arr);
                    }
                }
                ret_vec.extend(args);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::MkSeq {
                result: r,
                elems: inputs,
                seq_type: _,
                elem_type: _,
            } => {
                let mut ret_vec = vec![r];
                ret_vec.extend(inputs);
                ret_vec.into_iter()
            }
            Self::MkSeqOfBlob {
                result: r, blob, ..
            } => vec![r, blob].into_iter(),
            Self::MkRepeated {
                result: r,
                element,
                seq_type: _,
                count: _,
                elem_type: _,
            } => vec![r, element].into_iter(),
            Self::Select {
                result: a,
                cond: b,
                if_t: c,
                if_f: d,
            } => vec![a, b, c, d].into_iter(),
            Self::Not {
                result: r,
                value: v,
            }
            | Self::Spread {
                result: r,
                value: v,
                ..
            } => vec![r, v].into_iter(),
            Self::Unspread {
                result_odd: a,
                result_even: b,
                value: v,
                ..
            } => vec![a, b, v].into_iter(),
            Self::ToBits {
                result: r,
                value: v,
                endianness: _,
                count: _,
            } => vec![r, v].into_iter(),
            Self::ToRadix {
                result: r,
                value: v,
                radix,
                endianness: _,
                count: _,
            } => {
                let mut ret_vec = vec![r, v];
                match radix {
                    Radix::Bytes => {}
                    Radix::Dyn(radix) => {
                        ret_vec.push(radix);
                    }
                }
                ret_vec.into_iter()
            }
            Self::Rangecheck {
                value: val,
                max_bits: _,
            } => vec![val].into_iter(),
            Self::ReadGlobal {
                result: r,
                offset: _,
                result_type: _,
            } => vec![r].into_iter(),
            Self::TupleProj {
                result: r,
                tuple: t,
                idx: _,
            }
            | Self::TupleRefProj {
                result: r,
                tuple_ref: t,
                idx: _,
            } => vec![r, t].into_iter(),
            OpCode::MkTuple {
                result: r,
                elems: e,
                element_types: _,
            } => {
                let mut ret_vec = vec![r];
                ret_vec.extend(e);
                ret_vec.into_iter()
            }
            Self::Todo { results, .. } => {
                let ret_vec: Vec<&mut ValueId> = results.iter_mut().collect();
                ret_vec.into_iter()
            }
            Self::InitGlobal {
                global: _,
                value: v,
            } => vec![v].into_iter(),
            Self::DropGlobal { global: _ } => vec![].into_iter(),
            Self::Guard { condition, inner } => {
                let mut ret_vec = vec![condition];
                ret_vec.extend(inner.get_operands_mut());
                ret_vec.into_iter()
            }
        }
    }
}

// HLSSA TYPE ALIASES
// ================================================================================================

pub type HLFunction = Function<OpCode, Type>;
pub type HLBlock = Block<OpCode, Type>;

// CALL TARGET
// ================================================================================================

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CallTarget {
    Static(FunctionId),
    Dynamic(ValueId),
}

// BINARY ARITH OPERATION KIND
// ================================================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryArithOpKind {
    Add,
    Mul,
    Div,
    Sub,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Mod,
}

// COMPARISON KIND
// ================================================================================================

#[derive(Debug, Clone, Copy)]
pub enum CmpKind {
    Lt,
    Eq,
}

// SEQUENCE TYPE
// ================================================================================================

#[derive(Debug, Clone, Copy)]
pub enum SequenceTargetType {
    Array(usize),
    Slice,
    Tuple,
}

impl SequenceTargetType {
    pub fn of(&self, t: Type) -> Type {
        match self {
            SequenceTargetType::Array(len) => t.array_of(*len),
            SequenceTargetType::Slice => t.slice_of(),
            SequenceTargetType::Tuple => panic!("Tuple type requires multiple element types"),
        }
    }
}

impl Display for SequenceTargetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequenceTargetType::Array(len) => write!(f, "Array[{}]", len),
            SequenceTargetType::Slice => write!(f, "Slice"),
            SequenceTargetType::Tuple => write!(f, "Tuple"),
        }
    }
}

// CAST TARGET
// ================================================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CastTarget {
    Field,
    U(usize),
    I(usize),
    WitnessOf,
    /// Strips one `WitnessOf` wrapper.
    ValueOf,
    Nop,
    ArrayToSlice,
    /// Applies the inner cast to every element of an array or slice.
    /// Lowered to an explicit loop late, by `LowerMapCasts`; witness-only maps
    /// never reach codegen in the witgen pipeline (`StripWitnessOf` erases them).
    Map(Box<CastTarget>),
}

impl CastTarget {
    /// The cast target injecting witness wrappers to convert a value of type
    /// `src` into type `tgt`, or `None` when no conversion is needed (equal
    /// types, or types differing only in ways that share a runtime
    /// representation).
    ///
    /// Panics on conversions casts cannot express — in particular on
    /// witness *strips* (`WitnessOf(X) → X`): silently erasing witness-ness at
    /// a typed-slot boundary would drop the constraint connection, so such a
    /// request always indicates a witness-inference inconsistency. Strips are
    /// only legal for unconstrained call arguments, via [`Self::strip_conversion`].
    pub fn conversion(src: &Type, tgt: &Type) -> Option<CastTarget> {
        Self::conversion_impl(src, tgt, false)
    }

    /// The cast target stripping witness wrappers to convert a value of type
    /// `src` into type `tgt` (for unconstrained call arguments), or `None`
    /// when the types already match.
    pub fn strip_conversion(src: &Type, tgt: &Type) -> Option<CastTarget> {
        Self::conversion_impl(src, tgt, true)
    }

    fn conversion_impl(src: &Type, tgt: &Type, strip: bool) -> Option<CastTarget> {
        if src == tgt {
            return None;
        }
        match (&src.expr, &tgt.expr) {
            (TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_), TypeExpr::WitnessOf(_))
                if !strip =>
            {
                Some(CastTarget::WitnessOf)
            }
            (TypeExpr::WitnessOf(inner), TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_))
                if strip && inner.as_ref() == tgt =>
            {
                Some(CastTarget::ValueOf)
            }
            // Same runtime representation on both sides.
            (TypeExpr::WitnessOf(_), TypeExpr::WitnessOf(_)) if !strip => None,
            (TypeExpr::Ref(_), TypeExpr::Ref(_)) if !strip => None,
            (TypeExpr::Array(s, n), TypeExpr::Array(t, m)) => {
                assert_eq!(
                    n, m,
                    "array size mismatch in cast conversion: {src} -> {tgt}"
                );
                Self::conversion_impl(s, t, strip).map(|inner| CastTarget::Map(Box::new(inner)))
            }
            (TypeExpr::Slice(s), TypeExpr::Slice(t)) => {
                Self::conversion_impl(s, t, strip).map(|inner| CastTarget::Map(Box::new(inner)))
            }
            _ => panic!(
                "no cast target converts {:?} -> {:?} (strip: {})",
                src, tgt, strip
            ),
        }
    }

    /// The type a cast with this target produces for an input of `value_type`.
    pub fn result_type(&self, value_type: &Type) -> Type {
        match self {
            CastTarget::Field => {
                if value_type.is_witness_of() {
                    Type::witness_of(Type::field())
                } else {
                    Type::field()
                }
            }
            CastTarget::U(size) => {
                if value_type.is_witness_of() {
                    Type::witness_of(Type::u(*size))
                } else {
                    Type::u(*size)
                }
            }
            CastTarget::I(size) => {
                if value_type.is_witness_of() {
                    Type::witness_of(Type::i(*size))
                } else {
                    Type::i(*size)
                }
            }
            CastTarget::Nop => value_type.clone(),
            CastTarget::ArrayToSlice => match &value_type.expr {
                TypeExpr::Array(elem, _len) => elem.as_ref().clone().slice_of(),
                _ => panic!("ArrayToSlice cast on non-array type"),
            },
            CastTarget::WitnessOf => Type::witness_of(value_type.clone()),
            CastTarget::ValueOf => match &value_type.expr {
                TypeExpr::WitnessOf(inner) => inner.as_ref().clone(),
                _ => panic!("ValueOf cast on non-WitnessOf type {:?}", value_type),
            },
            CastTarget::Map(inner) => match &value_type.expr {
                TypeExpr::Array(elem, len) => inner.result_type(elem).array_of(*len),
                TypeExpr::Slice(elem) => inner.result_type(elem).slice_of(),
                _ => panic!("Map cast on non-sequence type {:?}", value_type),
            },
        }
    }

    /// Whether this cast only changes the witness representation of a value
    /// (injects or strips `WitnessOf` wrappers, possibly elementwise). After
    /// stripping `WitnessOf` from all types such casts are identities.
    pub fn is_witness_repr_only(&self) -> bool {
        match self {
            CastTarget::WitnessOf | CastTarget::ValueOf => true,
            CastTarget::Map(inner) => inner.is_witness_repr_only(),
            CastTarget::Field
            | CastTarget::U(_)
            | CastTarget::I(_)
            | CastTarget::Nop
            | CastTarget::ArrayToSlice => false,
        }
    }

    /// Whether this cast strips witness wrappers (`ValueOf`, possibly under
    /// `Map`s). Strips are only emitted for hint chains and unconstrained call
    /// arguments, both of which are dead by R1CS generation.
    pub fn is_value_of(&self) -> bool {
        match self {
            CastTarget::ValueOf => true,
            CastTarget::Map(inner) => inner.is_value_of(),
            CastTarget::Field
            | CastTarget::U(_)
            | CastTarget::I(_)
            | CastTarget::WitnessOf
            | CastTarget::Nop
            | CastTarget::ArrayToSlice => false,
        }
    }
}

impl Display for CastTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastTarget::Field => write!(f, "Field"),
            CastTarget::U(size) => write!(f, "u{}", size),
            CastTarget::I(size) => write!(f, "i{}", size),
            CastTarget::WitnessOf => write!(f, "WitnessOf"),
            CastTarget::ValueOf => write!(f, "ValueOf"),
            CastTarget::Nop => write!(f, "Nop"),
            CastTarget::ArrayToSlice => write!(f, "ArrayToSlice"),
            CastTarget::Map(inner) => write!(f, "Map({})", inner),
        }
    }
}

// ENDIANNESS
// ================================================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Endianness {
    Big,
    Little,
}

impl Display for Endianness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endianness::Big => write!(f, "big"),
            Endianness::Little => write!(f, "little"),
        }
    }
}

// SLICE OPERAND DIRECTION
// ================================================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceOpDir {
    Front,
    Back,
}

// CONSTANTS
// ================================================================================================

/// A compile-time-only sequence of constants used for long constant data.
/// Blobs are homogeneous: every element has type `elem_type`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Blob {
    pub elem_type: Type,
    pub elements: Vec<Constant>,
}

impl Blob {
    pub fn new(elem_type: Type, elements: Vec<Constant>) -> Self {
        Self {
            elem_type,
            elements,
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

/// The value type stored in the high-level SSA's constants side-table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    U(usize, u128),
    I(usize, u128),
    Field(ark_bn254::Fr),
    FnPtr(FunctionId),
    Blob(Blob),
}

// REFERENCE COUNTING OPS
// ================================================================================================

#[derive(Debug, Clone, Copy)]
pub enum RefCountOp {
    /// A reference count increment operation, bumping by the provided amount.
    Bump(usize),

    /// A reference count decrement operation (always by one).
    Drop,
}

// R1CS MATRICES
// ================================================================================================

#[derive(Debug, Clone, Copy)]
pub enum DMatrix {
    A,
    B,
    C,
}

// LOOKUP TARGET
// ================================================================================================

#[derive(Debug, Clone, Copy)]
pub enum LookupTarget<V> {
    Rangecheck(u8),
    DynRangecheck(V),
    Array(V),
    Spread(u8),
}

impl<V> LookupTarget<V> {
    /// Map the value payload (dynamic rangecheck bound or array operand), keeping the target kind.
    pub fn map<U>(&self, mut f: impl FnMut(&V) -> U) -> LookupTarget<U> {
        match self {
            LookupTarget::Rangecheck(bits) => LookupTarget::Rangecheck(*bits),
            LookupTarget::DynRangecheck(bound) => LookupTarget::DynRangecheck(f(bound)),
            LookupTarget::Array(array) => LookupTarget::Array(f(array)),
            LookupTarget::Spread(bits) => LookupTarget::Spread(*bits),
        }
    }
}

// RADIX
// ================================================================================================

#[derive(Debug, Clone, Copy)]
pub enum Radix<V> {
    Bytes,
    Dyn(V),
}
