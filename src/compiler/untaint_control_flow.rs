use std::collections::HashMap;

use tracing::{Level, instrument};

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    ssa::{
        BinaryArithOpKind, CallTarget, CastTarget, FunctionId, HLFunction, HLSSA, OpCode,
        Terminator, ValueId,
    },
    witness_info::{ConstantWitness, FunctionWitnessType, WitnessInfo},
    witness_type_inference::WitnessTypeInference,
};

pub struct UntaintControlFlow {}

/// Look up the witness level for a value, defaulting to Pure for values
/// not present in the witness type map (e.g., values created by WitnessCastInsertion).
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

impl UntaintControlFlow {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "UntaintControlFlow::run")]
    pub fn run(
        &mut self,
        mut ssa: HLSSA,
        witness_inference: &WitnessTypeInference,
        flow_analysis: &FlowAnalysis,
    ) -> HLSSA {
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            if let Some(function_wt) = witness_inference.try_get_function_witness_type(function_id)
            {
                let mut function = ssa.take_function(function_id);
                self.run_function(function_id, &mut function, function_wt, flow_analysis);
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
    ) {
        let cfg = flow_analysis.get_function_cfg(function_id);

        let cfg_witness_param = if matches!(function_wt.cfg_witness, WitnessInfo::Witness) {
            Some(function.add_parameter(function.get_entry_id(), Type::witness_of(Type::u(1))))
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
                if let OpCode::Cast {
                    result,
                    value,
                    target: CastTarget::WitnessOf,
                } = instr
                {
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
                    // Call keeps current taint-threading approach (not wrapped in Guard)
                    OpCode::Call {
                        results: ret,
                        function: CallTarget::Static(tgt),
                        mut args,
                        unconstrained,
                    } => {
                        if !unconstrained {
                            match block_taint {
                                Some(arg) => {
                                    args.push(arg);
                                }
                                None => {}
                            }
                        }
                        new_instructions.push(OpCode::Call {
                            results: ret,
                            function: CallTarget::Static(tgt),
                            args,
                            unconstrained,
                        });
                    }
                    OpCode::Call {
                        function: CallTarget::Dynamic(_),
                        ..
                    } => {
                        panic!("Dynamic call targets are not supported in untaint_control_flow")
                    }
                    // All non-Call ops: wrap in Guard when tainted
                    other => match block_taint {
                        Some(taint) => {
                            new_instructions.push(OpCode::Guard {
                                condition: taint,
                                inner: Box::new(other),
                            });
                        }
                        None => {
                            new_instructions.push(other);
                        }
                    },
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
                                                cond,
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
}
