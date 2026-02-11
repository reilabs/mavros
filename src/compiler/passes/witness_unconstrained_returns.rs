use std::collections::HashMap;
use crate::compiler::{ir::r#type::{Empty, Type, TypeExpr}, pass_manager::{DataPoint, Pass, PassInfo}, ssa::{BinaryArithOpKind, CallTarget, CastTarget, Function, FunctionId, OpCode, SeqType, TupleIdx, ValueId, SSA}};

pub struct WitnessUnconstrainedReturns {}

impl Pass<Empty> for WitnessUnconstrainedReturns {
    fn run(
        &self,
        ssa: &mut SSA<Empty>,
        _pass_manager: &crate::compiler::pass_manager::PassManager<Empty>,
    ) {
        Self::witness_unconstrained_returns(ssa);
    }

    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "witness_unconstrained_returns",
            needs: vec![DataPoint::CFG],
        }
    }
}

impl WitnessUnconstrainedReturns {
    pub fn new() -> Self {
        Self {}
    }

    /// Constrains return values from unconstrained function calls.
    /// For each unconstrained call, the return values are written to witness,
    /// range-checked (for non-Field types), and reconstructed.
    fn witness_unconstrained_returns(ssa: &mut SSA<Empty>) {
        let func_ids: Vec<_> = ssa.get_function_ids().collect();

        for func_id in func_ids {
            let is_unconstrained = ssa.get_function(func_id).is_unconstrained();
            if is_unconstrained {
                continue;
            }

            // Collect return types for all unconstrained calls in this function
            let called_fn_returns: HashMap<FunctionId, Vec<Type<Empty>>> = {
                let func = ssa.get_function(func_id);
                let mut map = HashMap::new();
                for (_, block) in func.get_blocks() {
                    for instr in block.get_instructions() {
                        if let OpCode::Call { function: CallTarget::Static(called_fn), is_unconstrained: true, .. } = instr {
                            if !map.contains_key(called_fn) {
                                map.insert(*called_fn, ssa.get_function(*called_fn).get_returns().to_vec());
                            }
                        }
                    }
                }
                map
            };

            if called_fn_returns.is_empty() {
                continue;
            }

            // Now process each block
            let func = ssa.get_function_mut(func_id);
            let block_ids: Vec<_> = func.get_blocks().map(|(id, _)| *id).collect();

            for block_id in block_ids {
                let old_instructions = func.get_block_mut(block_id).take_instructions();
                let mut new_instructions = Vec::new();

                for instr in old_instructions {
                    match instr {
                        OpCode::Call { results, function: CallTarget::Static(called_fn), args, is_unconstrained: true } => {
                            let return_types = called_fn_returns.get(&called_fn).unwrap();

                            // Create fresh raw IDs for all returns
                            let raw_results: Vec<_> = results.iter()
                                .map(|_| func.fresh_value())
                                .collect();

                            // Emit call with raw results
                            new_instructions.push(OpCode::Call {
                                results: raw_results.clone(),
                                function: CallTarget::Static(called_fn),
                                args,
                                is_unconstrained: true,
                            });

                            // Write each return value to witness and constrain
                            for ((orig_id, raw_id), typ) in results.iter().zip(raw_results.iter()).zip(return_types.iter()) {
                                Self::witness_and_constrain(
                                    *raw_id, *orig_id, typ,
                                    &mut new_instructions, func,
                                );
                            }
                        }
                        other => new_instructions.push(other),
                    }
                }

                func.get_block_mut(block_id).put_instructions(new_instructions);
            }
        }
    }

    /// Writes a value to witness and constrains it (range-check for non-Field types).
    /// Handles recursive decomposition for arrays and tuples.
    fn witness_and_constrain(
        raw_id: ValueId,
        result_id: ValueId,
        typ: &Type<Empty>,
        instructions: &mut Vec<OpCode<Empty>>,
        func: &mut Function<Empty>,
    ) {
        match &typ.expr {
            TypeExpr::Field => {
                // Write to witness, no range-check needed
                instructions.push(OpCode::WriteWitness {
                    result: Some(result_id),
                    value: raw_id,
                    witness_annotation: Empty,
                });
            }
            TypeExpr::U(size) => {
                // Cast to field, write to witness, range-check, cast back
                let as_field = func.fresh_value();
                instructions.push(OpCode::Cast {
                    result: as_field,
                    value: raw_id,
                    target: CastTarget::Field,
                });

                let witness_field = func.fresh_value();
                instructions.push(OpCode::WriteWitness {
                    result: Some(witness_field),
                    value: as_field,
                    witness_annotation: Empty,
                });

                if *size == 1 {
                    // Boolean constraint: x * (x - 1) = 0
                    let zero = func.push_field_const(ark_bn254::Fr::from(0));
                    let one = func.push_field_const(ark_bn254::Fr::from(1));
                    let x_sub_1 = func.fresh_value();
                    let x_times_x_sub_1 = func.fresh_value();
                    instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Sub,
                        result: x_sub_1,
                        lhs: witness_field,
                        rhs: one,
                    });
                    instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Mul,
                        result: x_times_x_sub_1,
                        lhs: witness_field,
                        rhs: x_sub_1,
                    });
                    instructions.push(OpCode::AssertEq {
                        lhs: x_times_x_sub_1,
                        rhs: zero,
                    });
                } else {
                    instructions.push(OpCode::Rangecheck {
                        value: witness_field,
                        max_bits: *size,
                    });
                }

                instructions.push(OpCode::Cast {
                    result: result_id,
                    value: witness_field,
                    target: CastTarget::U(*size),
                });
            }
            TypeExpr::Array(inner, size) => {
                let mut elem_ids = Vec::new();

                for i in 0..*size {
                    let index = func.push_u_const(32, i as u128);
                    let child_raw = func.fresh_value();
                    instructions.push(OpCode::ArrayGet {
                        result: child_raw,
                        array: raw_id,
                        index,
                    });

                    let child_result = func.fresh_value();
                    Self::witness_and_constrain(
                        child_raw, child_result, inner,
                        instructions, func,
                    );
                    elem_ids.push(child_result);
                }

                instructions.push(OpCode::MkSeq {
                    result: result_id,
                    elems: elem_ids,
                    seq_type: SeqType::Array(*size),
                    elem_type: *inner.clone(),
                });
            }
            TypeExpr::Tuple(element_types) => {
                let mut elem_ids = Vec::new();
                let mut elem_types = Vec::new();

                for (i, elem_type) in element_types.iter().enumerate() {
                    let child_raw = func.fresh_value();
                    instructions.push(OpCode::TupleProj {
                        result: child_raw,
                        tuple: raw_id,
                        idx: TupleIdx::Static(i),
                    });

                    let child_result = func.fresh_value();
                    Self::witness_and_constrain(
                        child_raw, child_result, elem_type,
                        instructions, func,
                    );
                    elem_ids.push(child_result);
                    elem_types.push(elem_type.clone());
                }

                instructions.push(OpCode::MkTuple {
                    result: result_id,
                    elems: elem_ids,
                    element_types: elem_types,
                });
            }
            _ => todo!("witness_and_constrain not implemented for type: {:?}", typ)
        }
    }
}
