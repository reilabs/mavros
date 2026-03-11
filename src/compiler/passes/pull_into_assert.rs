use std::collections::HashMap;

use crate::compiler::{
    analysis::types::{FunctionTypeInfo, TypeInfo},
    flow_analysis::FlowAnalysis,
    ir::r#type::TypeExpr,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{BinaryArithOpKind, HLSSA, Instruction, OpCode, ValueId},
};

pub struct PullIntoAssert {}

pub struct PulledProduct {
    lhs: ValueId,
    rhs: ValueId,
}

impl Pass for PullIntoAssert {
    fn name(&self) -> &'static str {
        "pull_into_assert"
    }
    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }
    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }
}

impl PullIntoAssert {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let function_type_info = type_info.get_function(*function_id);
            let mut uses: HashMap<ValueId, usize> = HashMap::new();
            let mut defs: HashMap<ValueId, OpCode> = HashMap::new();

            for (_, block) in function.get_blocks() {
                for instruction in block.get_instructions() {
                    for input in instruction.get_inputs() {
                        *uses.entry(*input).or_insert(0) += 1;
                    }
                    for result in instruction.get_results() {
                        defs.insert(*result, instruction.clone());
                    }
                }
            }

            let mut new_blocks = HashMap::new();

            for (block_id, mut block) in function.take_blocks() {
                let mut new_instructions = Vec::new();
                for instruction in block.take_instructions().into_iter() {
                    match instruction {
                        OpCode::AssertEq { lhs, rhs } => {
                            let mut pull = self.try_pull(lhs, &uses, &defs, function_type_info);
                            let mut other_op = rhs;
                            if pull.is_none() {
                                pull = self.try_pull(rhs, &uses, &defs, function_type_info);
                                other_op = lhs;
                            }

                            let pull = match pull {
                                Some(pull) => pull,
                                None => {
                                    new_instructions.push(instruction.clone());
                                    continue;
                                }
                            };

                            new_instructions.push(OpCode::AssertR1C {
                                a: pull.lhs,
                                b: pull.rhs,
                                c: other_op,
                            });
                        }
                        _ => {
                            new_instructions.push(instruction.clone());
                        }
                    }
                }
                block.put_instructions(new_instructions);
                new_blocks.insert(block_id, block);
            }

            function.put_blocks(new_blocks);
        }
    }

    fn try_pull(
        &self,
        value: ValueId,
        uses: &HashMap<ValueId, usize>,
        defs: &HashMap<ValueId, OpCode>,
        function_type_info: &FunctionTypeInfo,
    ) -> Option<PulledProduct> {
        if *uses.get(&value).unwrap_or(&0) > 1 {
            return None;
        }
        let def = defs.get(&value)?;
        match def {
            // TODO: we should also pull further, skipping pure multiplications and shoving
            // them into the constants or R1CS constraints
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result,
                lhs,
                rhs,
            } => {
                // Only pull field-typed muls into AssertR1C; uint muls
                // need range-checking and can't be directly constrained.
                let result_type = function_type_info.get_value_type(*result);
                match result_type.strip_witness().expr {
                    TypeExpr::Field => Some(PulledProduct {
                        lhs: *lhs,
                        rhs: *rhs,
                    }),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
