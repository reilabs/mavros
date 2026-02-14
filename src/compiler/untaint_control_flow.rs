use std::collections::HashMap;

use tracing::{Level, instrument};

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{BinaryArithOpKind, CallTarget, CastTarget, Function, FunctionId, OpCode, SSA, Terminator, ValueId},
    witness_info::{ConstantWitness, FunctionWitnessType, WitnessInfo},
    witness_type_inference::WitnessTypeInference,
};

pub struct UntaintControlFlow {}

/// Look up the witness level for a value, defaulting to Pure for values
/// not present in the witness type map (e.g., values created by WitnessCastInsertion).
fn get_witness_or_pure(function_wt: &FunctionWitnessType, v: crate::compiler::ssa::ValueId) -> ConstantWitness {
    function_wt
        .value_witness_types
        .get(&v)
        .map(|wt| wt.toplevel_info())
        .unwrap_or(ConstantWitness::Pure)
}

impl UntaintControlFlow {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run")]
    pub fn run(
        &mut self,
        mut ssa: SSA,
        witness_inference: &WitnessTypeInference,
        flow_analysis: &FlowAnalysis,
    ) -> SSA {
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            let function_wt = witness_inference.get_function_witness_type(function_id);
            let mut function = ssa.take_function(function_id);
            self.run_function(function_id, &mut function, function_wt, flow_analysis);
            ssa.put_function(function_id, function);
        }

        // Post-process: wrapper_main's entry params should be plain Field/U types,
        // not WitnessOf. The VM writes concrete values to these positions.
        // Insert WriteWitness to bridge Field → WitnessOf(Field) before passing
        // to original_main.
        Self::fix_main_entry_params(&mut ssa);

        ssa
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run_function", level = Level::DEBUG, fields(function = function.get_name()))]
    fn run_function(
        &mut self,
        function_id: FunctionId,
        function: &mut Function,
        function_wt: &FunctionWitnessType,
        flow_analysis: &FlowAnalysis,
    ) {
        let cfg = flow_analysis.get_function_cfg(function_id);

        let cfg_witness_param = if matches!(
            function_wt.cfg_witness,
            WitnessInfo::Witness
        ) {
            Some(
                function.add_parameter(function.get_entry_id(), Type::witness_of(Type::u(1))),
            )
        } else {
            None
        };

        // Build map of Cast { target: WitnessOf } results → pre-cast values.
        // WitnessCastInsertion inserts these casts at Jmp boundaries to match
        // merge block param types. We strip them from Select branches so that
        // the Select typing rule derives WitnessOf from the condition instead.
        let mut witness_cast_strip: HashMap<ValueId, ValueId> = HashMap::new();
        for (_, block) in function.get_blocks() {
            for instr in block.get_instructions() {
                if let OpCode::Cast { result, value, target: CastTarget::WitnessOf } = instr {
                    witness_cast_strip.insert(*result, *value);
                }
            }
        }

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
                match instruction {
                    OpCode::BinaryArithOp { .. }
                    | OpCode::Cmp { .. }
                    | OpCode::MkSeq { .. }
                    | OpCode::MkTuple { .. }
                    | OpCode::Cast { .. }
                    | OpCode::Truncate { .. }
                    | OpCode::Not { .. }
                    | OpCode::Rangecheck { .. }
                    | OpCode::ToBits { .. }
                    | OpCode::ToRadix { .. }
                    | OpCode::ReadGlobal { .. }
                    | OpCode::Select { .. }
                    | OpCode::WriteWitness { .. }
                    | OpCode::FreshWitness { .. }
                    | OpCode::Constrain { .. }
                    | OpCode::NextDCoeff { .. }
                    | OpCode::BumpD { .. }
                    | OpCode::MemOp { .. }
                    | OpCode::MulConst { .. }
                    | OpCode::Lookup { .. }
                    | OpCode::DLookup { .. }
                    | OpCode::TupleProj { .. }
                    | OpCode::AssertR1C { .. } => {
                        new_instructions.push(instruction);
                    }
                    OpCode::AssertEq { lhs, rhs } => {
                        match block_taint {
                            Some(taint) => {
                                let new_rhs = function.fresh_value();
                                new_instructions.push(OpCode::Select {
                                    result: new_rhs,
                                    cond: taint,
                                    if_t: rhs,
                                    if_f: lhs,
                                });
                                new_instructions.push(OpCode::AssertEq {
                                    lhs: lhs,
                                    rhs: new_rhs,
                                })
                            }
                            None => new_instructions.push(instruction),
                        }
                    }
                    OpCode::Store { ptr, value: v } => {
                        let ptr_wt = get_witness_or_pure(function_wt, ptr);
                        // writes to dynamic ptr not supported
                        assert_eq!(ptr_wt, ConstantWitness::Pure);

                        match block_taint {
                            Some(taint) => {
                                let old_value = function.fresh_value();
                                new_instructions.push(OpCode::Load {
                                    result: old_value,
                                    ptr: ptr,
                                });

                                let new_value = function.fresh_value();
                                new_instructions.push(OpCode::Select {
                                    result: new_value,
                                    cond: taint,
                                    if_t: v,
                                    if_f: old_value,
                                });

                                new_instructions.push(OpCode::Store {
                                    ptr: ptr,
                                    value: new_value,
                                });
                            }
                            None => new_instructions.push(instruction),
                        }
                    }
                    OpCode::Load { result: _, ptr } => {
                        let ptr_wt = get_witness_or_pure(function_wt, ptr);
                        // reads from dynamic ptr not supported
                        assert_eq!(ptr_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::ArrayGet {
                        result: _,
                        array: arr,
                        index: _,
                    } => {
                        let arr_wt = get_witness_or_pure(function_wt, arr);
                        // dynamic array access not supported
                        assert_eq!(arr_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::ArraySet {
                        result: _,
                        array: arr,
                        index: idx,
                        value: _,
                    } => {
                        let arr_wt = get_witness_or_pure(function_wt, arr);
                        let idx_wt = get_witness_or_pure(function_wt, idx);
                        // dynamic array access not supported
                        assert_eq!(arr_wt, ConstantWitness::Pure);
                        assert_eq!(idx_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::SlicePush {
                        dir: _,
                        result: _,
                        slice: sl,
                        values: _,
                    } => {
                        let slice_wt = get_witness_or_pure(function_wt, sl);
                        // Slice must always be Pure witness
                        assert_eq!(slice_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::SliceLen {
                        result: _,
                        slice: sl,
                    } => {
                        let slice_wt = get_witness_or_pure(function_wt, sl);
                        // Slice must always be Pure witness
                        assert_eq!(slice_wt, ConstantWitness::Pure);
                        new_instructions.push(instruction);
                    }
                    OpCode::Alloc { .. } => {
                        new_instructions.push(instruction);
                    }

                    OpCode::Call {
                        results: ret,
                        function: CallTarget::Static(tgt),
                        mut args,
                    } => {
                        match block_taint {
                            Some(arg) => {
                                args.push(arg);
                            }
                            None => {}
                        }
                        new_instructions.push(OpCode::Call {
                            results: ret,
                            function: CallTarget::Static(tgt),
                            args: args,
                        });
                    }
                    OpCode::Call { function: CallTarget::Dynamic(_), .. } => {
                        panic!("Dynamic call targets are not supported in untaint_control_flow")
                    }

                    OpCode::Todo { payload, results, result_types } => {
                        new_instructions.push(OpCode::Todo {
                            payload,
                            results,
                            result_types,
                        });
                    }

                    OpCode::InitGlobal { .. } | OpCode::DropGlobal { .. } => {
                        new_instructions.push(instruction);
                    }
                    _ => {
                        panic!("Unhandled instruction {:?}", instruction);
                    }
                }
            }

            match block.get_terminator().cloned() {
                Some(Terminator::JmpIf(cond, if_true, if_false)) => {
                    let cond_wt = get_witness_or_pure(function_wt, cond);
                    match cond_wt {
                        ConstantWitness::Pure => {}
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

                            // If one of the branches is empty, we just jump to the other and
                            // there's no need to merge any values – this is purely side-effectful,
                            // which the instruction rewrites will handle.
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

                                // We remove the parameters from the merge block – they will be un-phi-fied
                                let merge_params = function.get_block_mut(merge).take_parameters();

                                for (_, typ) in &merge_params {
                                    match typ {
                                        Type {
                                            expr: TypeExpr::Array(_, _),
                                            ..
                                        } => {
                                            panic!("TODO: Witness array not supported yet")
                                        }
                                        Type {
                                            expr: TypeExpr::Ref(_),
                                            ..
                                        } => {
                                            panic!("TODO: Witness ref not supported yet")
                                        }
                                        _ => {}
                                    }
                                }

                                let args_passed_from_lhs = match function
                                    .get_block_mut(out_true_block)
                                    .take_terminator()
                                {
                                    Some(Terminator::Jmp(_, args)) => args,
                                    _ => panic!(
                                        "Impossible – out jump must be a JMP, otherwise the join point wouldn't be a join point"
                                    ),
                                };

                                // Jump straight to the false block
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
                                    for ((res, _), (lhs, rhs)) in merge_params.iter().zip(
                                        args_passed_from_lhs
                                            .iter()
                                            .zip(args_passed_from_rhs.iter()),
                                    ) {
                                        // Strip WitnessOf casts from branches — the Select
                                        // typing rule derives WitnessOf from the condition.
                                        let lhs = witness_cast_strip.get(lhs).unwrap_or(lhs);
                                        let rhs = witness_cast_strip.get(rhs).unwrap_or(rhs);
                                        function.get_block_mut(merger_block).push_instruction(
                                            OpCode::Select {
                                                result: *res,
                                                cond: cond,
                                                if_t: *lhs,
                                                if_f: *rhs,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            };

            block.put_instructions(new_instructions);
            function.put_block(block_id, block);
        }
    }

    /// Fix wrapper_main entry params: strip WitnessOf from param types
    /// and insert WriteWitness instructions to convert Field → WitnessOf(Field).
    /// This is needed because the VM writes concrete Field values (4 limbs) to
    /// entry params, but WitnessOf(Field) is pointer-sized (1 slot).
    fn fix_main_entry_params(ssa: &mut SSA) {
        let main_id = ssa.get_main_id();
        let main_fn = ssa.get_function_mut(main_id);
        let entry_id = main_fn.get_entry_id();
        let entry_block = main_fn.get_block_mut(entry_id);

        let old_params = entry_block.take_parameters();
        let old_instructions = entry_block.take_instructions();

        let mut new_params = Vec::new();
        let mut write_witness_instructions = Vec::new();
        let mut replacements = ValueReplacements::new();

        for (value_id, typ) in &old_params {
            if typ.is_witness_of() {
                // Strip WitnessOf from param type — entry params are concrete values
                let inner_type = typ.strip_witness();
                new_params.push((*value_id, inner_type));

                // Create WriteWitness to bridge Field → WitnessOf(Field)
                let witness_val = main_fn.fresh_value();
                write_witness_instructions.push(OpCode::WriteWitness {
                    result: Some(witness_val),
                    value: *value_id,
                });
                replacements.insert(*value_id, witness_val);
            } else {
                new_params.push((*value_id, typ.clone()));
            }
        }

        // Rebuild entry block: params + WriteWitness instructions + original instructions
        let entry_block = main_fn.get_block_mut(entry_id);

        let mut new_instructions = write_witness_instructions;
        for mut instruction in old_instructions {
            replacements.replace_instruction(&mut instruction);
            new_instructions.push(instruction);
        }

        entry_block.put_parameters(new_params);
        // Also fix the terminator (e.g., return args)
        replacements.replace_terminator(entry_block.get_terminator_mut());
        entry_block.put_instructions(new_instructions);
    }
}
