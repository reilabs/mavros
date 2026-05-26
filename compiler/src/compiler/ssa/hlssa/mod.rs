//! The high-level SSA representation used in the compiler and its associated types.

pub mod builder;
pub mod type_system;

use itertools::Itertools;
use std::collections::HashMap;
use std::fmt::Display;

use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    passes::fix_double_jumps::ValueReplacements,
    ssa::{Block, ConstantsDisplay, Function, FunctionId, Instruction, SSA, ValueId},
};
pub use type_system::{Type, TypeExpr};

// HLSSA
// ================================================================================================

/// The high-level SSA is designed for domain-level analysis without concretizing runtime details.
pub type HLSSA = SSA<OpCode, Type, Constants>;

impl HLSSA {
    pub fn new() -> Self {
        Self::with_main("main".to_string(), Constants::default())
    }

    /// Returns the canonical `ValueId` for `value`, allocating a fresh one and recording it in the
    /// constants table if `value` has not been interned yet. Otherwise returns the existing id.
    pub fn intern_const(&mut self, value: ConstValue) -> ValueId {
        if let Some(&id) = self.const_storage().get_by_right(&value) {
            return id;
        }
        let id = self.fresh_value();
        self.const_storage_mut().insert(id, value);
        id
    }

    /// Look up a constant by `ValueId`. Returns `None` if `id` is not a constant.
    pub fn get_const(&self, id: ValueId) -> Option<&ConstValue> {
        self.const_storage().get_by_left(&id)
    }

    /// Folds `other` into `self`, allocating fresh identifiers as it goes and re-interning
    /// constants so duplicates collapse.
    ///
    /// Returns the source-to-destination `FunctionId` map so callers can locate the merged-in
    /// functions; constant- and value-ID remappings stay internal.
    ///
    /// Unreachable blocks in `other` (not visited by a dominator-order walk from each function's
    /// entry block) are dropped. Run dead-code elimination on `other` first if they must be
    /// preserved.
    pub fn merge(&mut self, other: HLSSA) -> HashMap<FunctionId, FunctionId> {
        let global_offset = self.num_globals();
        if global_offset > 0 || !other.get_global_types().is_empty() {
            let mut combined = self.get_global_types().to_vec();
            combined.extend(other.get_global_types().iter().cloned());
            self.set_global_types(combined);
        }
        if self.get_globals_init_fn().is_some() && other.get_globals_init_fn().is_some() {
            panic!("ICE: HLSSA::merge cannot compose two globals_init_fns");
        }
        if self.get_globals_deinit_fn().is_some() && other.get_globals_deinit_fn().is_some() {
            panic!("ICE: HLSSA::merge cannot compose two globals_deinit_fns");
        }

        let mut fn_map: HashMap<FunctionId, FunctionId> = HashMap::new();
        let src_fn_ids: Vec<FunctionId> = other
            .iter_functions()
            .map(|(id, _)| *id)
            .sorted_by_key(|id| id.0)
            .collect();
        for src_fn_id in &src_fn_ids {
            let name = other.get_function(*src_fn_id).get_name().to_string();
            let dst_fn_id = self.add_function(name);
            fn_map.insert(*src_fn_id, dst_fn_id);
        }

        // The re-interned constants are used to seed a value replacements table for downstream
        // replacement in the function.
        let mut replacements = ValueReplacements::new();
        let mut const_entries: Vec<(ValueId, ConstValue)> = other
            .const_storage()
            .iter()
            .map(|(id, cv)| (*id, cv.clone()))
            .collect();
        const_entries.sort_by_key(|(id, _)| id.0);
        for (src_id, cv) in const_entries {
            let rewritten = match cv {
                ConstValue::FnPtr(fid) => ConstValue::FnPtr(fn_map[&fid]),
                other_cv => other_cv,
            };
            let new_id = self.intern_const(rewritten);
            replacements.insert(src_id, new_id);
        }

        // The functions are walked in dominator order to ensure that we can do replacements in a
        // single pass, rather than two.
        let other_flow = FlowAnalysis::run(&other);
        let mut other = other;
        for src_fn_id in &src_fn_ids {
            let cfg = other_flow.get_function_cfg(*src_fn_id);
            let src_fn = other.take_function(*src_fn_id);
            let (mut new_fn, mut src_blocks, returns) = src_fn.prepare_rebuild();
            for ret in returns {
                new_fn.add_return_type(ret);
            }

            for block_id in cfg.get_domination_pre_order() {
                let Some(mut block) = src_blocks.remove(&block_id) else {
                    continue;
                };

                let old_params = block.take_parameters();
                let mut new_params = Vec::with_capacity(old_params.len());
                for (old_param, ty) in old_params {
                    let new_param = self.fresh_value();
                    replacements.insert(old_param, new_param);
                    new_params.push((new_param, ty));
                }
                block.put_parameters(new_params);

                let old_instructions = block.take_instructions();
                let mut new_instructions = Vec::with_capacity(old_instructions.len());
                for mut instr in old_instructions {
                    // Allocate fresh IDs for every result so `replace_instruction` finds them.
                    let old_results: Vec<ValueId> = instr.get_results().copied().collect();
                    for old in old_results {
                        let new = self.fresh_value();
                        replacements.insert(old, new);
                    }
                    // Lookup/DLookup are strange, so we have to handle these manually.
                    match &instr {
                        OpCode::Lookup { results, .. } | OpCode::DLookup { results, .. } => {
                            for old in results.clone() {
                                let new = self.fresh_value();
                                replacements.insert(old, new);
                            }
                        }
                        _ => {}
                    }

                    replacements.replace_instruction(&mut instr);
                    remap_static_calls(&mut instr, &fn_map);
                    if global_offset > 0 {
                        shift_globals(&mut instr, global_offset);
                    }

                    new_instructions.push(instr);
                }
                block.put_instructions(new_instructions);

                if let Some(mut term) = block.take_terminator() {
                    replacements.replace_terminator(&mut term);
                    block.set_terminator(term);
                }

                new_fn.put_block(block_id, block);
            }

            self.put_function(fn_map[src_fn_id], new_fn);
        }

        if self.get_globals_init_fn().is_none() {
            if let Some(fid) = other.get_globals_init_fn() {
                self.set_globals_init_fn(fn_map[&fid]);
            }
        }
        if self.get_globals_deinit_fn().is_none() {
            if let Some(fid) = other.get_globals_deinit_fn() {
                self.set_globals_deinit_fn(fn_map[&fid]);
            }
        }

        fn_map
    }
}

/// Rewrites `FunctionId`s embedded in static `Call` targets (and inside `Guard`'s inner op)
/// using `fn_map`. `ValueReplacements` only rewrites `ValueId`s, so this complements it.
fn remap_static_calls(instr: &mut OpCode, fn_map: &HashMap<FunctionId, FunctionId>) {
    match instr {
        OpCode::Call {
            function: CallTarget::Static(fid),
            ..
        } => {
            *fid = fn_map[fid];
        }
        OpCode::Guard { inner, .. } => remap_static_calls(inner, fn_map),
        _ => {}
    }
}

/// Shifts global indices in `ReadGlobal`/`InitGlobal`/`DropGlobal` (and inside `Guard`) so that
/// `other`'s globals land in the appended section of `self`'s globals table.
fn shift_globals(instr: &mut OpCode, offset: usize) {
    match instr {
        OpCode::ReadGlobal { offset: o, .. } => *o += offset as u64,
        OpCode::InitGlobal { global, .. } => *global += offset,
        OpCode::DropGlobal { global } => *global += offset,
        OpCode::Guard { inner, .. } => shift_globals(inner, offset),
        _ => {}
    }
}

// CONSTANT STORAGE
// ================================================================================================

/// Constant storage for the high-level SSA: a bidirectional map between `ValueId` and `ConstValue`,
/// enforcing one canonical `ValueId` per distinct constant value.
pub type Constants = bimap::BiHashMap<ValueId, ConstValue>;

/// Render a single `ConstValue` as the right-hand-side of a constants-table entry, matching the
/// syntax previously used for the `OpCode::Const` instruction.
pub fn display_const(value: &ConstValue, func_name: &dyn Fn(FunctionId) -> String) -> String {
    match value {
        ConstValue::U(size, val) => format!("u_const({}, {})", size, val),
        ConstValue::I(size, val) => format!("i_const({}, {})", size, val),
        ConstValue::Field(val) => format!("field_const({})", val),
        ConstValue::FnPtr(fn_id) => {
            format!("fn_ptr_const({}@{})", func_name(*fn_id), fn_id.0)
        }
    }
}

impl ConstantsDisplay for Constants {
    fn display_constants(&self, func_name: &dyn Fn(FunctionId) -> String) -> String {
        if self.is_empty() {
            return String::new();
        }
        let entries = self
            .iter()
            .sorted_by_key(|(id, _)| id.0)
            .map(|(id, cv)| format!("  v{} = {}", id.0, display_const(cv, func_name)))
            .join("\n");
        format!("constants:\n{}", entries)
    }
}

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
    Truncate {
        result: ValueId,
        value: ValueId,
        to_bits: usize,
        from_bits: usize,
    },
    SExt {
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
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
    MkRepeated {
        result: ValueId,
        element: ValueId,
        seq_type: SequenceTargetType,
        count: usize,
        elem_type: Type,
    },
    Alloc {
        result: ValueId,
        elem_type: Type,
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
    ValueOf {
        result: ValueId,
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
        keys: Vec<ValueId>,
        results: Vec<ValueId>,
        flag: ValueId,
    },
    DLookup {
        target: LookupTarget<ValueId>,
        keys: Vec<ValueId>,
        results: Vec<ValueId>,
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
            OpCode::Alloc {
                result,
                elem_type: typ,
            } => format!("v{}{} = alloc({})", result.0, annotate_value(*result), typ),
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
            OpCode::Lookup {
                target,
                keys,
                results,
                flag,
            } => {
                let keys_str = keys.iter().map(|v| format!("v{}", v.0)).join(", ");
                let results_str = results.iter().map(|v| format!("v{}", v.0)).join(", ");
                let target_str = match target {
                    LookupTarget::Rangecheck(n) => format!("rngchk({})", n),
                    LookupTarget::DynRangecheck(v) => format!("rngchk(_ < v{})", v.0),
                    LookupTarget::Array(arr) => format!("v{}", arr.0),
                    LookupTarget::Spread(n) => format!("spread({})", n),
                };
                format!(
                    "constrain_lookup({}, ({}) => ({}), flag=v{})",
                    target_str, keys_str, results_str, flag.0
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
            OpCode::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
                let keys_str = keys.iter().map(|v| format!("v{}", v.0)).join(", ");
                let results_str = results.iter().map(|v| format!("v{}", v.0)).join(", ");
                let target_str = match target {
                    LookupTarget::Rangecheck(n) => format!("rngchk({})", n),
                    LookupTarget::DynRangecheck(v) => format!("rngchk(_ < v{})", v.0),
                    LookupTarget::Array(arr) => format!("v{}", arr.0),
                    LookupTarget::Spread(n) => format!("spread({})", n),
                };
                format!(
                    "∂lookup({}, ({}) => ({}), flag=v{})",
                    target_str, keys_str, results_str, flag.0
                )
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
            OpCode::Truncate {
                result,
                value,
                to_bits: out_bits,
                from_bits: in_bits,
            } => {
                format!(
                    "v{}{} = truncate v{} from {} bits to {} bits",
                    result.0,
                    annotate_value(*result),
                    value.0,
                    in_bits,
                    out_bits
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
            OpCode::Not { result, value } => {
                format!("v{}{} = ~v{}", result.0, annotate_value(*result), value.0)
            }
            OpCode::ValueOf { result, value } => {
                format!(
                    "v{}{} = value_of v{}",
                    result.0,
                    annotate_value(*result),
                    value.0
                )
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
                elem_type: _,
            }
            | Self::FreshWitness {
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
            | Self::Truncate {
                result: _,
                value: c,
                to_bits: _,
                from_bits: _,
            }
            | Self::SExt {
                result: _,
                value: c,
                from_bits: _,
                to_bits: _,
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
            }
            | Self::ValueOf {
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
            Self::Lookup {
                target,
                keys,
                results,
                flag,
            }
            | Self::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
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
                ret_vec.extend(keys);
                ret_vec.extend(results);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::TupleProj {
                result: _,
                tuple,
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
            | Self::MkRepeated { result: r, .. }
            | Self::Select { result: r, .. }
            | Self::Cast { result: r, .. }
            | Self::Truncate { result: r, .. }
            | Self::SExt { result: r, .. }
            | Self::MulConst { result: r, .. }
            | Self::NextDCoeff { result: r }
            | Self::TupleProj { result: r, .. }
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
            Self::Not { result: r, .. }
            | Self::ValueOf { result: r, .. }
            | Self::Spread { result: r, .. } => vec![r].into_iter(),
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
            | Self::MkRepeated { result: r, .. }
            | Self::Select { result: r, .. }
            | Self::Cast { result: r, .. }
            | Self::Truncate { result: r, .. }
            | Self::SExt { result: r, .. }
            | Self::MulConst { result: r, .. }
            | Self::NextDCoeff { result: r }
            | Self::TupleProj { result: r, .. }
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
            Self::Not { result: r, .. }
            | Self::ValueOf { result: r, .. }
            | Self::Spread { result: r, .. } => vec![r].into_iter(),
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
                elem_type: _,
            }
            | Self::FreshWitness {
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
            | Self::Truncate {
                result: _,
                value: c,
                to_bits: _,
                from_bits: _,
            }
            | Self::SExt {
                result: _,
                value: c,
                from_bits: _,
                to_bits: _,
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
            }
            | Self::ValueOf {
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
            Self::Lookup {
                target,
                keys,
                results,
                flag,
            }
            | Self::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
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
                ret_vec.extend(keys);
                ret_vec.extend(results);
                ret_vec.push(flag);
                ret_vec.into_iter()
            }
            Self::TupleProj {
                result: _,
                tuple,
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
                elem_type: _,
            }
            | Self::MemOp { kind: _, value: r }
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
            Self::Truncate {
                result: a,
                value: b,
                to_bits: _,
                from_bits: _,
            } => vec![a, b].into_iter(),
            Self::SExt {
                result: a,
                value: b,
                from_bits: _,
                to_bits: _,
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
            Self::Lookup {
                target,
                keys,
                results,
                flag,
            }
            | Self::DLookup {
                target,
                keys,
                results,
                flag,
            } => {
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
                ret_vec.extend(keys);
                ret_vec.extend(results);
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
            | Self::ValueOf {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CastTarget {
    Field,
    U(usize),
    I(usize),
    WitnessOf,
    Nop,
    ArrayToSlice,
}

impl Display for CastTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastTarget::Field => write!(f, "Field"),
            CastTarget::U(size) => write!(f, "u{}", size),
            CastTarget::I(size) => write!(f, "i{}", size),
            CastTarget::WitnessOf => write!(f, "WitnessOf"),
            CastTarget::Nop => write!(f, "Nop"),
            CastTarget::ArrayToSlice => write!(f, "ArrayToSlice"),
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

// CONST VALUES
// ================================================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstValue {
    U(usize, u128),
    I(usize, u128),
    Field(ark_bn254::Fr),
    FnPtr(FunctionId),
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

// RADIX
// ================================================================================================

#[derive(Debug, Clone, Copy)]
pub enum Radix<V> {
    Bytes,
    Dyn(V),
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::Terminator;

    /// Builds a tiny HLSSA with a configurable constant and a Call site for testing merge.
    /// Returns: (ssa, main_id, callee_id, the U(32, _) constant ValueId, the FnPtr constant ValueId).
    fn build_fixture(
        callee_name: &str,
        const_val: u128,
    ) -> (HLSSA, FunctionId, FunctionId, ValueId, ValueId) {
        let mut ssa = HLSSA::new();
        let main_id = ssa.get_main_id();
        let callee_id = ssa.add_function(callee_name.to_string());

        let u_const = ssa.intern_const(ConstValue::U(32, const_val));
        let fn_ptr = ssa.intern_const(ConstValue::FnPtr(callee_id));

        // main: entry block does `r = u_const + u_const`, then calls callee, then returns r.
        let r = ssa.fresh_value();
        let call_result = ssa.fresh_value();
        let main = ssa.get_function_mut(main_id);
        let entry = main.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: r,
            lhs: u_const,
            rhs: u_const,
        });
        entry.push_instruction(OpCode::Call {
            results: vec![call_result],
            function: CallTarget::Static(callee_id),
            args: vec![r],
            unconstrained: false,
        });
        entry.set_terminator(Terminator::Return(vec![r]));

        // callee: returns its single parameter unchanged.
        let p = ssa.fresh_value();
        let callee = ssa.get_function_mut(callee_id);
        callee.add_return_type(Type::u(32));
        let entry = callee.get_entry_mut();
        entry.push_parameter(p, Type::u(32));
        entry.set_terminator(Terminator::Return(vec![p]));

        (ssa, main_id, callee_id, u_const, fn_ptr)
    }

    #[test]
    fn merge_returns_function_map_for_every_source_function() {
        let (target, _, _, _, _) = build_fixture("callee_a", 7);
        let (source, src_main, src_callee, _, _) = build_fixture("callee_b", 7);
        let mut target = target;
        let fn_map = target.merge(source);

        assert!(fn_map.contains_key(&src_main));
        assert!(fn_map.contains_key(&src_callee));
        assert_eq!(fn_map.len(), 2);
        // The destination IDs must not collide with anything already in target.
        for (src_id, dst_id) in &fn_map {
            assert_ne!(src_id, dst_id);
            assert_ne!(target.get_main_id(), *dst_id);
        }
    }

    #[test]
    fn merge_deduplicates_shared_constants() {
        // Both fixtures intern U(32, 7); after merge target should still have exactly 3 constants:
        // U(32, 7), FnPtr(target_callee), FnPtr(source_callee) — the U is shared, the FnPtrs differ
        // because they reference different functions.
        let (mut target, _, _, _, _) = build_fixture("callee_a", 7);
        let (source, _, _, _, _) = build_fixture("callee_b", 7);
        let before = target.const_storage().len();
        let _ = target.merge(source);
        assert_eq!(before, 2); // U(32,7) + FnPtr(target_callee)
        assert_eq!(target.const_storage().len(), 3);

        // Both FnPtr constants must point at distinct destination functions.
        let fn_ptrs: Vec<FunctionId> = target
            .const_storage()
            .iter()
            .filter_map(|(_, cv)| match cv {
                ConstValue::FnPtr(fid) => Some(*fid),
                _ => None,
            })
            .collect();
        assert_eq!(fn_ptrs.len(), 2);
        assert_ne!(fn_ptrs[0], fn_ptrs[1]);
    }

    #[test]
    fn merge_remaps_static_call_target() {
        let (mut target, _, _, _, _) = build_fixture("callee_a", 7);
        let (source, src_main, src_callee, _, _) = build_fixture("callee_b", 11);
        let fn_map = target.merge(source);
        let merged_main = target.get_function(fn_map[&src_main]);
        let call_target = merged_main
            .get_entry()
            .get_instructions()
            .find_map(|instr| match instr {
                OpCode::Call {
                    function: CallTarget::Static(fid),
                    ..
                } => Some(*fid),
                _ => None,
            })
            .expect("merged main should contain a static Call");
        assert_eq!(call_target, fn_map[&src_callee]);
    }

    #[test]
    fn merge_allocates_fresh_value_ids() {
        let (mut target, _, _, _, _) = build_fixture("callee_a", 7);
        let value_bound_before = target.value_num_bound();

        let (source, src_main, _, _, _) = build_fixture("callee_b", 11);
        let fn_map = target.merge(source);

        // Every operand in the merged main must reference a value either already in target
        // (a constant — bound < before) or one freshly allocated by merge (>= before).
        // What must hold strictly: none of the operands point at a ValueId that didn't exist
        // before merge but isn't a constant in the new table.
        let merged_main = target.get_function(fn_map[&src_main]);
        for instr in merged_main.get_entry().get_instructions() {
            for v in instr.get_inputs().chain(instr.get_results()) {
                assert!(
                    (v.0 as usize) < target.value_num_bound(),
                    "ValueId {} out of bounds",
                    v.0
                );
                let is_constant = target.get_const(*v).is_some();
                let is_fresh = (v.0 as usize) >= value_bound_before;
                assert!(
                    is_constant || is_fresh,
                    "v{} is neither a constant nor a freshly-allocated id",
                    v.0
                );
            }
        }
    }
}
