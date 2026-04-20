use std::collections::HashMap;

use tracing::{Level, instrument};

use crate::compiler::{
    analysis::types::{FunctionTypeInfo, TypeInfo, Types},
    block_builder::{HLEmitter, HLInstrBuilder},
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    ssa::{
        BinaryArithOpKind, BlockId, CallTarget, CastTarget, FunctionId, HLBlock, HLFunction, HLSSA,
        OpCode, SeqType, Terminator, ValueId,
    },
    witness_info::{ConstantWitness, FunctionWitnessType, WitnessInfo, WitnessType},
    witness_type_inference::WitnessTypeInference,
};

pub struct UntaintControlFlow {}

/// Look up the witness level for a value, defaulting to Pure for values
/// not present in the witness type map (e.g., values created after type inference).
fn get_witness_or_pure(
    function_wt: &FunctionWitnessType,
    v: crate::compiler::ssa::ValueId,
) -> ConstantWitness {
    function_wt
        .value_witness_types
        .get(&v)
        .map(|wt| wt.toplevel_info())
        .unwrap_or(ConstantWitness::Pure)
}

/// Push an instruction, wrapping in Guard if block is tainted.
fn maybe_guard(instrs: &mut Vec<OpCode>, taint: Option<ValueId>, instr: OpCode) {
    match taint {
        Some(taint) => instrs.push(OpCode::Guard {
            condition: taint,
            inner: Box::new(instr),
        }),
        None => instrs.push(instr),
    }
}

impl UntaintControlFlow {
    pub fn new() -> Self {
        Self {}
    }

    // -----------------------------------------------------------------------
    // Step 1: Type application — bake WitnessOf into SSA types
    //
    // Walks every function that has witness type info and rewrites SSA types
    // (block params, instruction result types, return types) to include
    // WitnessOf wrappers where witness inference determined a value is witness-
    // dependent. This must run before cast insertion / linearization so that
    // the type info pass can see the WitnessOf types.
    // -----------------------------------------------------------------------

    #[instrument(skip_all, name = "UntaintControlFlow::apply_types")]
    fn apply_types(&self, ssa: HLSSA, witness_inference: &WitnessTypeInference) -> HLSSA {
        let (mut result_ssa, functions, old_global_types) = ssa.prepare_rebuild();
        result_ssa.set_global_types(old_global_types);

        for (function_id, function) in functions.into_iter() {
            if let Some(function_wt) = witness_inference.try_get_function_witness_type(function_id)
            {
                let new_function = self.apply_types_to_function(function, function_wt);
                result_ssa.put_function(function_id, new_function);
            } else {
                result_ssa.put_function(function_id, function);
            }
        }

        result_ssa
    }

    fn apply_types_to_function(
        &self,
        function: HLFunction,
        function_wt: &FunctionWitnessType,
    ) -> HLFunction {
        let (mut function, blocks, returns) = function.prepare_rebuild();

        for (block_id, mut block) in blocks.into_iter() {
            let mut new_block = HLBlock::empty();

            let mut new_parameters = Vec::new();
            for (value_id, typ) in block.take_parameters() {
                let wt = function_wt.get_value_witness_type(value_id);
                new_parameters.push((value_id, apply_witness_type(typ, wt)));
            }
            new_block.put_parameters(new_parameters);

            let mut new_instructions = Vec::<OpCode>::new();
            for instruction in block.take_instructions() {
                let new = match instruction {
                    OpCode::Alloc {
                        result: r,
                        elem_type: l,
                    } => {
                        let r_wt = function_wt.get_value_witness_type(r);
                        let child = r_wt.child_witness_type().unwrap();
                        let child_typ = apply_witness_type(l, &child);
                        OpCode::Alloc {
                            result: r,
                            elem_type: child_typ,
                        }
                    }
                    OpCode::FreshWitness { .. } => instruction,
                    OpCode::MkSeq {
                        result: r,
                        elems: l,
                        seq_type: stp,
                        elem_type: tp,
                    } => {
                        let r_wt = function_wt
                            .get_value_witness_type(r)
                            .child_witness_type()
                            .unwrap();
                        OpCode::MkSeq {
                            result: r,
                            elems: l,
                            seq_type: stp,
                            elem_type: apply_witness_type(tp, &r_wt),
                        }
                    }
                    OpCode::MkTuple {
                        result: r,
                        elems: l,
                        element_types: tps,
                    } => {
                        let r_wt = function_wt.get_value_witness_type(r);
                        let child_wts = if let WitnessType::Tuple(_, children) = r_wt {
                            children
                        } else {
                            panic!("MkTuple result should have Tuple witness type")
                        };
                        OpCode::MkTuple {
                            result: r,
                            elems: l,
                            element_types: tps
                                .iter()
                                .zip(child_wts.iter())
                                .map(|(tp, wt)| apply_witness_type(tp.clone(), wt))
                                .collect(),
                        }
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: tp,
                    } => OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: tp,
                    },
                    OpCode::Todo {
                        payload,
                        results,
                        result_types,
                    } => OpCode::Todo {
                        payload,
                        results,
                        result_types,
                    },
                    other => other,
                };
                new_instructions.push(new);
            }
            new_block.put_instructions(new_instructions);
            new_block.set_terminator(block.take_terminator().unwrap());
            function.put_block(block_id, new_block);
        }

        for (ret, ret_wt) in returns.into_iter().zip(function_wt.returns_witness.iter()) {
            let ret_typ = apply_witness_type(ret, ret_wt);
            function.add_return_type(ret_typ);
        }

        function
    }

    // -----------------------------------------------------------------------
    // Step 2: Cast insertion + control flow linearization
    //
    // After types are baked in and flow/type analysis is recomputed:
    //  - Linearizes witness-conditional branches: witness JmpIf is replaced
    //    by unconditional Jmp + Select at merge points; instructions in
    //    tainted blocks are wrapped in Guard.
    //  - Inserts WitnessOf casts at typed-slot boundaries (MkSeq elems,
    //    ArraySet value, SlicePush values, Store value, Select operands,
    //    Jmp args, Return values) where the actual type doesn't match the
    //    expected slot type.
    //  - Pushes cfg_witness arg to constrained calls; strips WitnessOf from
    //    unconstrained call args via ValueOf.
    // -----------------------------------------------------------------------

    #[instrument(skip_all, name = "UntaintControlFlow::run")]
    pub fn run(&mut self, ssa: HLSSA, witness_inference: &WitnessTypeInference) -> HLSSA {
        // Step 1: bake WitnessOf into SSA types
        let mut ssa = self.apply_types(ssa, witness_inference);

        // Recompute flow + type info (types changed in step 1)
        let flow_analysis = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow_analysis);

        // Step 2: cast insertion + control flow linearization
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            if let Some(function_wt) = witness_inference.try_get_function_witness_type(function_id)
            {
                let func_type_info = if type_info.has_function(function_id) {
                    Some(type_info.get_function(function_id))
                } else {
                    None
                };
                let mut function = ssa.take_function(function_id);
                self.run_function(
                    function_id,
                    &mut function,
                    function_wt,
                    &flow_analysis,
                    func_type_info,
                );
                ssa.put_function(function_id, function);
            }
        }

        ssa
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run_function", level = Level::DEBUG, fields(function = function.get_name()))]
    fn run_function(
        &mut self,
        function_id: FunctionId,
        function: &mut HLFunction,
        function_wt: &FunctionWitnessType,
        flow_analysis: &FlowAnalysis,
        type_info: Option<&FunctionTypeInfo>,
    ) {
        let cfg = flow_analysis.get_function_cfg(function_id);

        let cfg_witness_param = if matches!(function_wt.cfg_witness, WitnessInfo::Witness) {
            Some(function.add_parameter(function.get_entry_id(), Type::witness_of(Type::u(1))))
        } else {
            None
        };

        // Collect block param types for Jmp cast insertion
        let block_param_types: HashMap<BlockId, Vec<Type>> = function
            .get_blocks()
            .map(|(bid, block)| {
                let types = block.get_parameters().map(|(_, tp)| tp.clone()).collect();
                (*bid, types)
            })
            .collect();

        let return_types: Vec<Type> = function.get_returns().to_vec();

        let mut block_taint_vars = HashMap::new();
        for (block_id, _) in function.get_blocks() {
            block_taint_vars.insert(*block_id, cfg_witness_param.clone());
        }

        for block_id in cfg.get_blocks_bfs() {
            let mut block = function.take_block(block_id);
            let block_taint = block_taint_vars.get(&block_id).unwrap().clone();

            let old_instructions = block.take_instructions();
            let mut new_instructions = Vec::new();

            for instruction in old_instructions {
                self.process_instruction(
                    instruction,
                    function,
                    type_info,
                    block_taint,
                    &mut new_instructions,
                );
            }

            // Handle terminator
            match block.get_terminator().cloned() {
                Some(Terminator::JmpIf(cond, if_true, if_false)) => {
                    let cond_wt = get_witness_or_pure(function_wt, cond);
                    match cond_wt {
                        ConstantWitness::Pure => {
                            // Pure JmpIf: insert casts at Jmp boundaries in branch blocks
                            // (handled when those blocks are processed)
                        }
                        ConstantWitness::Witness => {
                            let child_block_taint = match block_taint {
                                Some(tnt) => {
                                    let result_val = function.fresh_value();
                                    new_instructions.push(OpCode::BinaryArithOp {
                                        kind: BinaryArithOpKind::And,
                                        result: result_val,
                                        lhs: tnt,
                                        rhs: cond,
                                    });
                                    result_val
                                }
                                None => cond,
                            };
                            let body = cfg.get_if_body(block_id);
                            for block_id in body {
                                block_taint_vars.insert(block_id, Some(child_block_taint));
                            }

                            let merge = cfg.get_merge_point(block_id);

                            if merge == if_true {
                                block.set_terminator(Terminator::Jmp(if_false, vec![]));
                            } else if merge == if_false {
                                block.set_terminator(Terminator::Jmp(if_true, vec![]));
                            } else {
                                block.set_terminator(Terminator::Jmp(if_true, vec![]));

                                if merge == function.get_entry_id() {
                                    panic!(
                                        "TODO: jump back into entry not supported yet. Is it even possible?"
                                    )
                                }

                                let jumps = cfg.get_jumps_into_merge_from_branch(if_true, merge);
                                if jumps.len() != 1 {
                                    panic!(
                                        "TODO: handle multiple jumps into merge {:?} {:?} {:?} {:?}",
                                        block_id, if_true, merge, jumps
                                    );
                                }
                                let out_true_block = jumps[0];

                                let merge_params = function.get_block_mut(merge).take_parameters();

                                let args_passed_from_lhs = match function
                                    .get_block_mut(out_true_block)
                                    .take_terminator()
                                {
                                    Some(Terminator::Jmp(_, args)) => args,
                                    _ => panic!(
                                        "Impossible – out jump must be a JMP, otherwise the join point wouldn't be a join point"
                                    ),
                                };

                                function
                                    .get_block_mut(out_true_block)
                                    .set_terminator(Terminator::Jmp(if_false, vec![]));

                                let jumps = cfg.get_jumps_into_merge_from_branch(if_false, merge);
                                if jumps.len() != 1 {
                                    panic!(
                                        "TODO: handle multiple jumps into merge {:?} {:?} {:?} {:?}",
                                        block_id, if_false, merge, jumps
                                    );
                                }
                                let out_false_block = jumps[0];
                                let args_passed_from_rhs = match function
                                    .get_block_mut(out_false_block)
                                    .take_terminator()
                                {
                                    Some(Terminator::Jmp(_, args)) => args,
                                    _ => panic!(
                                        "Impossible – out jump must be a JMP, otherwise the join point wouldn't be a join point"
                                    ),
                                };

                                let merger_block = function.add_block();
                                function
                                    .get_block_mut(out_false_block)
                                    .set_terminator(Terminator::Jmp(merger_block, vec![]));
                                function
                                    .get_block_mut(merger_block)
                                    .set_terminator(Terminator::Jmp(merge, vec![]));

                                if args_passed_from_lhs.len() > 0 {
                                    let mut instrs = Vec::new();
                                    {
                                        let mut builder =
                                            HLInstrBuilder::new(function, &mut instrs);
                                        for ((res, typ), (lhs, rhs)) in merge_params.iter().zip(
                                            args_passed_from_lhs
                                                .iter()
                                                .zip(args_passed_from_rhs.iter()),
                                        ) {
                                            let lhs_type = type_info
                                                .map(|ti| ti.get_value_type(*lhs).clone())
                                                .unwrap_or_else(|| typ.clone());
                                            let rhs_type = type_info
                                                .map(|ti| ti.get_value_type(*rhs).clone())
                                                .unwrap_or_else(|| typ.clone());
                                            emit_merge_select(
                                                &mut builder,
                                                cond,
                                                *lhs,
                                                *rhs,
                                                Some(*res),
                                                typ,
                                                &lhs_type,
                                                &rhs_type,
                                            );
                                        }
                                    }
                                    for instr in instrs {
                                        function
                                            .get_block_mut(merger_block)
                                            .push_instruction(instr);
                                    }
                                }
                            }
                        }
                    }
                }
                Some(Terminator::Jmp(target, args)) => {
                    // Insert casts at Jmp boundaries
                    if let (Some(ti), Some(param_types)) =
                        (type_info, block_param_types.get(&target))
                    {
                        let mut cast_instrs = Vec::new();
                        let new_args: Vec<_> = {
                            let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                            args.iter()
                                .zip(param_types.iter())
                                .map(|(arg, expected_type)| {
                                    convert_if_needed(*arg, expected_type, ti, &mut builder)
                                })
                                .collect()
                        };
                        for instr in cast_instrs {
                            maybe_guard(&mut new_instructions, block_taint, instr);
                        }
                        block.set_terminator(Terminator::Jmp(target, new_args));
                    }
                }
                Some(Terminator::Return(values)) => {
                    if let Some(ti) = type_info {
                        let mut cast_instrs = Vec::new();
                        let new_values: Vec<_> = {
                            let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                            values
                                .iter()
                                .zip(return_types.iter())
                                .map(|(val, expected_type)| {
                                    convert_if_needed(*val, expected_type, ti, &mut builder)
                                })
                                .collect()
                        };
                        for instr in cast_instrs {
                            maybe_guard(&mut new_instructions, block_taint, instr);
                        }
                        block.set_terminator(Terminator::Return(new_values));
                    }
                }
                None => {}
            };

            block.put_instructions(new_instructions);
            function.put_block(block_id, block);
        }
    }

    /// Process a single instruction: apply cast insertion, then Guard-wrap if tainted.
    fn process_instruction(
        &self,
        instruction: OpCode,
        function: &mut HLFunction,
        type_info: Option<&FunctionTypeInfo>,
        block_taint: Option<ValueId>,
        new_instructions: &mut Vec<OpCode>,
    ) {
        match instruction {
            // -- Constrained Call: push cfg_witness arg --
            OpCode::Call {
                results: ret,
                function: CallTarget::Static(tgt),
                mut args,
                unconstrained: false,
            } => {
                if let Some(arg) = block_taint {
                    args.push(arg);
                }
                new_instructions.push(OpCode::Call {
                    results: ret,
                    function: CallTarget::Static(tgt),
                    args,
                    unconstrained: false,
                });
            }
            // -- Unconstrained Call: strip WitnessOf from args --
            OpCode::Call {
                results,
                function: CallTarget::Static(tgt),
                args,
                unconstrained: true,
            } => {
                if let Some(ti) = type_info {
                    let mut cast_instrs = Vec::new();
                    let new_args: Vec<_> = {
                        let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                        args.into_iter()
                            .map(|arg| {
                                let arg_type = ti.get_value_type(arg);
                                let pure_type = arg_type.strip_all_witness();
                                if *arg_type != pure_type {
                                    emit_strip_witness(arg, arg_type, &pure_type, &mut builder)
                                } else {
                                    arg
                                }
                            })
                            .collect()
                    };
                    for instr in cast_instrs {
                        maybe_guard(new_instructions, block_taint, instr);
                    }
                    new_instructions.push(OpCode::Call {
                        results,
                        function: CallTarget::Static(tgt),
                        args: new_args,
                        unconstrained: true,
                    });
                } else {
                    new_instructions.push(OpCode::Call {
                        results,
                        function: CallTarget::Static(tgt),
                        args,
                        unconstrained: true,
                    });
                }
            }
            OpCode::Call {
                function: CallTarget::Dynamic(_),
                ..
            } => {
                panic!("Dynamic call targets are not supported in untaint_control_flow")
            }
            // -- Cast insertion for MkSeq --
            OpCode::MkSeq {
                result: r,
                elems: vs,
                seq_type: s,
                elem_type: ref tp,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let target_elem_type = tp.clone();
                let mut cast_instrs = Vec::new();
                let new_vs: Vec<_> = {
                    let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                    vs.iter()
                        .map(|v| convert_if_needed(*v, &target_elem_type, ti, &mut builder))
                        .collect()
                };
                for instr in cast_instrs {
                    maybe_guard(new_instructions, block_taint, instr);
                }
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::MkSeq {
                        result: r,
                        elems: new_vs,
                        seq_type: s,
                        elem_type: target_elem_type,
                    },
                );
            }
            // -- Cast insertion for ArraySet --
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let array_type = ti.get_value_type(array);
                let expected_elem_type = match &array_type.expr {
                    TypeExpr::Array(inner, _) => inner.as_ref().clone(),
                    TypeExpr::Slice(inner) => inner.as_ref().clone(),
                    _ => panic!("ArraySet on non-array type"),
                };
                let mut cast_instrs = Vec::new();
                let converted = {
                    let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                    convert_if_needed(value, &expected_elem_type, ti, &mut builder)
                };
                for instr in cast_instrs {
                    maybe_guard(new_instructions, block_taint, instr);
                }
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::ArraySet {
                        result,
                        array,
                        index,
                        value: converted,
                    },
                );
            }
            // -- Cast insertion for SlicePush --
            OpCode::SlicePush {
                dir,
                result,
                slice,
                values,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let slice_type = ti.get_value_type(slice);
                let expected_elem_type = match &slice_type.expr {
                    TypeExpr::Slice(inner) => inner.as_ref().clone(),
                    _ => panic!("SlicePush on non-slice type"),
                };
                let mut cast_instrs = Vec::new();
                let new_values: Vec<_> = {
                    let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                    values
                        .iter()
                        .map(|v| convert_if_needed(*v, &expected_elem_type, ti, &mut builder))
                        .collect()
                };
                for instr in cast_instrs {
                    maybe_guard(new_instructions, block_taint, instr);
                }
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::SlicePush {
                        dir,
                        result,
                        slice,
                        values: new_values,
                    },
                );
            }
            // -- Cast insertion for Store --
            OpCode::Store { ptr, value } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let ptr_type = ti.get_value_type(ptr);
                let target_type = ptr_type.get_pointed();
                let mut cast_instrs = Vec::new();
                let converted = {
                    let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                    convert_if_needed(value, &target_type, ti, &mut builder)
                };
                for instr in cast_instrs {
                    maybe_guard(new_instructions, block_taint, instr);
                }
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::Store {
                        ptr,
                        value: converted,
                    },
                );
            }
            // -- Cast insertion for Select --
            OpCode::Select {
                result: r,
                cond,
                if_t,
                if_f,
            } if type_info.is_some() => {
                let ti = type_info.unwrap();
                let if_t_type = ti.get_value_type(if_t);
                let if_f_type = ti.get_value_type(if_f);
                let target_type = if_t_type.get_arithmetic_result_type(if_f_type);
                let mut cast_instrs = Vec::new();
                let (new_if_t, new_if_f) = {
                    let mut builder = HLInstrBuilder::new(function, &mut cast_instrs);
                    let t = convert_if_needed(if_t, &target_type, ti, &mut builder);
                    let f = convert_if_needed(if_f, &target_type, ti, &mut builder);
                    (t, f)
                };
                for instr in cast_instrs {
                    maybe_guard(new_instructions, block_taint, instr);
                }
                maybe_guard(
                    new_instructions,
                    block_taint,
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t: new_if_t,
                        if_f: new_if_f,
                    },
                );
            }
            // -- All other non-Call ops: Guard-wrap when tainted --
            other => maybe_guard(new_instructions, block_taint, other),
        }
    }
}

// ---------------------------------------------------------------------------
// Cast insertion helpers
// ---------------------------------------------------------------------------

fn convert_if_needed(
    value: ValueId,
    target_type: &Type,
    type_info: &FunctionTypeInfo,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    let value_type = type_info.get_value_type(value);
    if *value_type == *target_type {
        return value;
    }
    emit_value_conversion(value, value_type, target_type, builder)
}

/// Convert a value from source_type to target_type. Uses unrolled element-wise
/// conversion for arrays (no loop blocks needed), so this is safe inside guarded regions.
fn emit_value_conversion(
    value: ValueId,
    source_type: &Type,
    target_type: &Type,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    match (&source_type.expr, &target_type.expr) {
        // Scalar: Field → WitnessOf(Field), U(n) → WitnessOf(U(n))
        (TypeExpr::Field, TypeExpr::WitnessOf(_))
        | (TypeExpr::U(_), TypeExpr::WitnessOf(_))
        | (TypeExpr::I(_), TypeExpr::WitnessOf(_)) => builder.cast_to_witness_of(value),
        // Array element conversion (unrolled, no loop blocks)
        (TypeExpr::Array(src_inner, src_size), TypeExpr::Array(tgt_inner, tgt_size)) => {
            assert_eq!(
                src_size, tgt_size,
                "Array size mismatch in witness cast insertion"
            );
            emit_unrolled_array_conversion(value, src_inner, tgt_inner, *src_size, builder)
        }
        // Tuple: decompose, per-field recursive, recompose
        (TypeExpr::Tuple(src_fields), TypeExpr::Tuple(tgt_fields)) => {
            assert_eq!(
                src_fields.len(),
                tgt_fields.len(),
                "Tuple field count mismatch in witness cast insertion"
            );
            let mut converted_elems = vec![];
            for (i, (src_ft, tgt_ft)) in src_fields.iter().zip(tgt_fields.iter()).enumerate() {
                let proj = builder.tuple_proj(value, i);
                let converted = emit_value_conversion(proj, src_ft, tgt_ft, builder);
                converted_elems.push(converted);
            }
            builder.mk_tuple(converted_elems, tgt_fields.clone())
        }
        // WitnessOf→WitnessOf: identity (same runtime repr)
        (TypeExpr::WitnessOf(_), TypeExpr::WitnessOf(_)) => value,
        _ => panic!(
            "witness cast insertion: unsupported conversion {:?} -> {:?}",
            source_type, target_type
        ),
    }
}

/// Unrolled element-wise array conversion. Avoids creating loop blocks,
/// so it is safe to use inside guarded (tainted) regions.
fn emit_unrolled_array_conversion(
    source_array: ValueId,
    src_elem_type: &Type,
    tgt_elem_type: &Type,
    size: usize,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    let mut elems = Vec::with_capacity(size);
    for i in 0..size {
        let idx = builder.u_const(32, i as u128);
        let elem = builder.array_get(source_array, idx);
        let converted = emit_value_conversion(elem, src_elem_type, tgt_elem_type, builder);
        elems.push(converted);
    }
    builder.mk_seq(elems, SeqType::Array(size), tgt_elem_type.clone())
}

/// Recursively strip WitnessOf from a value (for unconstrained call args).
/// Uses unrolled element-wise conversion for arrays.
fn emit_strip_witness(
    value: ValueId,
    source_type: &Type,
    target_type: &Type,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    match (&source_type.expr, &target_type.expr) {
        _ if source_type == target_type => value,
        // WitnessOf(X) → X: emit ValueOf
        (TypeExpr::WitnessOf(inner), _) => {
            let unwrapped = builder.value_of(value);
            emit_strip_witness(unwrapped, inner, target_type, builder)
        }
        // Array: unrolled element-wise strip
        (TypeExpr::Array(src_inner, src_size), TypeExpr::Array(tgt_inner, tgt_size)) => {
            assert_eq!(src_size, tgt_size);
            let mut elems = Vec::with_capacity(*src_size);
            for i in 0..*src_size {
                let idx = builder.u_const(32, i as u128);
                let elem = builder.array_get(value, idx);
                let converted = emit_strip_witness(elem, src_inner, tgt_inner, builder);
                elems.push(converted);
            }
            builder.mk_seq(elems, SeqType::Array(*src_size), *tgt_inner.clone())
        }
        // Tuple: decompose, per-field recursive, recompose
        (TypeExpr::Tuple(src_fields), TypeExpr::Tuple(tgt_fields)) => {
            assert_eq!(src_fields.len(), tgt_fields.len());
            let mut converted_elems = vec![];
            for (i, (sf, tf)) in src_fields.iter().zip(tgt_fields.iter()).enumerate() {
                let proj = builder.tuple_proj(value, i);
                let converted = emit_strip_witness(proj, sf, tf, builder);
                converted_elems.push(converted);
            }
            builder.mk_tuple(converted_elems, tgt_fields.clone())
        }
        _ => panic!(
            "emit_strip_witness: unsupported conversion {:?} -> {:?}",
            source_type, target_type
        ),
    }
}

/// Emit selects for merge point values, handling type conversion between
/// branch values and the expected merge param type. For arrays, does unrolled
/// element-wise select + cast. For scalars, emits Select with optional cast.
fn emit_merge_select(
    builder: &mut HLInstrBuilder<'_>,
    cond: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    result: Option<ValueId>,
    result_type: &Type,
    lhs_type: &Type,
    rhs_type: &Type,
) -> ValueId {
    match &result_type.expr {
        TypeExpr::Array(result_elem_type, size) => {
            let lhs_elem_type = match &lhs_type.expr {
                TypeExpr::Array(e, _) => e.as_ref(),
                _ => panic!(
                    "emit_merge_select: expected array for lhs, got {:?}",
                    lhs_type
                ),
            };
            let rhs_elem_type = match &rhs_type.expr {
                TypeExpr::Array(e, _) => e.as_ref(),
                _ => panic!(
                    "emit_merge_select: expected array for rhs, got {:?}",
                    rhs_type
                ),
            };
            let mut elems = Vec::with_capacity(*size);
            for i in 0..*size {
                let idx = builder.u_const(32, i as u128);
                let lhs_elem = builder.array_get(lhs, idx);
                let rhs_elem = builder.array_get(rhs, idx);
                let selected = emit_merge_select(
                    builder,
                    cond,
                    lhs_elem,
                    rhs_elem,
                    None,
                    result_elem_type,
                    lhs_elem_type,
                    rhs_elem_type,
                );
                elems.push(selected);
            }
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::MkSeq {
                result,
                elems,
                seq_type: SeqType::Array(*size),
                elem_type: *result_elem_type.clone(),
            });
            result
        }
        TypeExpr::Tuple(result_fields) => {
            let lhs_fields = match &lhs_type.expr {
                TypeExpr::Tuple(f) => f,
                _ => panic!(
                    "emit_merge_select: expected tuple for lhs, got {:?}",
                    lhs_type
                ),
            };
            let rhs_fields = match &rhs_type.expr {
                TypeExpr::Tuple(f) => f,
                _ => panic!(
                    "emit_merge_select: expected tuple for rhs, got {:?}",
                    rhs_type
                ),
            };
            let mut elems = Vec::with_capacity(result_fields.len());
            for (i, ((rf, lf), rhsf)) in result_fields
                .iter()
                .zip(lhs_fields.iter())
                .zip(rhs_fields.iter())
                .enumerate()
            {
                let lhs_field = builder.tuple_proj(lhs, i);
                let rhs_field = builder.tuple_proj(rhs, i);
                let selected =
                    emit_merge_select(builder, cond, lhs_field, rhs_field, None, rf, lf, rhsf);
                elems.push(selected);
            }
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::MkTuple {
                result,
                elems,
                element_types: result_fields.clone(),
            });
            result
        }
        TypeExpr::WitnessOf(_) => {
            // Cast operands to WitnessOf if they aren't already
            let lhs = if !lhs_type.is_witness_of() {
                builder.cast_to_witness_of(lhs)
            } else {
                lhs
            };
            let rhs = if !rhs_type.is_witness_of() {
                builder.cast_to_witness_of(rhs)
            } else {
                rhs
            };
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => {
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.push(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Ref(_) => panic!("Witness select on Ref type not supported"),
        TypeExpr::Slice(_) => panic!("Witness select on Slice type not supported"),
        TypeExpr::Function => panic!("Witness select on Function type not supported"),
    }
}

// ---------------------------------------------------------------------------
// Type application helper
// ---------------------------------------------------------------------------

fn apply_witness_type(typ: Type, wt: &WitnessType) -> Type {
    match (typ.expr, wt) {
        (TypeExpr::Field, WitnessType::Scalar(info)) => {
            let base = Type::field();
            if info.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::U(size), WitnessType::Scalar(info)) => {
            let base = Type::u(size);
            if info.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::I(size), WitnessType::Scalar(info)) => {
            let base = Type::i(size);
            if info.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::Array(inner, size), WitnessType::Array(top, inner_wt)) => {
            let base = apply_witness_type(*inner, inner_wt.as_ref()).array_of(size);
            if top.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::Slice(inner), WitnessType::Array(top, inner_wt)) => {
            let base = apply_witness_type(*inner, inner_wt.as_ref()).slice_of();
            if top.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::Ref(inner), WitnessType::Ref(top, inner_wt)) => {
            let base = apply_witness_type(*inner, inner_wt.as_ref()).ref_of();
            if top.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (TypeExpr::Tuple(child_types), WitnessType::Tuple(top, child_wts)) => {
            let base = Type::tuple_of(
                child_types
                    .iter()
                    .zip(child_wts.iter())
                    .map(|(child_type, child_wt)| apply_witness_type(child_type.clone(), child_wt))
                    .collect(),
            );
            if top.is_witness() {
                Type::witness_of(base)
            } else {
                base
            }
        }
        (tp, wt) => panic!("Unexpected type {:?} with witness type {:?}", tp, wt),
    }
}
