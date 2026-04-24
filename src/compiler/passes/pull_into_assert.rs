use std::collections::HashMap;

use crate::compiler::{
    analysis::types::{FunctionTypeInfo, TypeInfo},
    flow_analysis::FlowAnalysis,
    ir::r#type::TypeExpr,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{BinaryArithOpKind, CmpKind, HLSSA, Instruction, OpCode, ValueId},
};

pub struct PullIntoAssert {}

struct PulledProduct {
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

            // Fixpoint: keep iterating until no more transformations apply
            loop {
                let mut changed = false;
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
                            OpCode::Assert { value } => {
                                let expanded = self.expand_assert(
                                    value,
                                    &uses,
                                    &defs,
                                    function_type_info,
                                );
                                if let Some(expanded) = expanded {
                                    changed = true;
                                    new_instructions.extend(expanded);
                                } else {
                                    new_instructions.push(instruction);
                                }
                            }
                            OpCode::AssertCmp {
                                kind: CmpKind::Eq,
                                lhs,
                                rhs,
                            } => {
                                let pull =
                                    self.try_pull_mul(lhs, &uses, &defs, function_type_info);
                                let (pull, other_op) = match pull {
                                    Some(p) => (p, rhs),
                                    None => {
                                        match self.try_pull_mul(
                                            rhs,
                                            &uses,
                                            &defs,
                                            function_type_info,
                                        ) {
                                            Some(p) => (p, lhs),
                                            None => {
                                                new_instructions.push(OpCode::AssertCmp {
                                                    kind: CmpKind::Eq,
                                                    lhs,
                                                    rhs,
                                                });
                                                continue;
                                            }
                                        }
                                    }
                                };
                                changed = true;
                                new_instructions.push(OpCode::AssertR1C {
                                    a: pull.lhs,
                                    b: pull.rhs,
                                    c: other_op,
                                });
                            }
                            _ => {
                                new_instructions.push(instruction);
                            }
                        }
                    }
                    block.put_instructions(new_instructions);
                    new_blocks.insert(block_id, block);
                }

                function.put_blocks(new_blocks);

                if !changed {
                    break;
                }
            }
        }
    }

    /// Try to expand an `Assert { value }` into a simpler form.
    /// Returns `Some(instructions)` if a transformation applies, `None` otherwise.
    fn expand_assert(
        &self,
        value: ValueId,
        uses: &HashMap<ValueId, usize>,
        defs: &HashMap<ValueId, OpCode>,
        function_type_info: &FunctionTypeInfo,
    ) -> Option<Vec<OpCode>> {
        // Only pull single-use definitions
        if *uses.get(&value).unwrap_or(&0) > 1 {
            return None;
        }
        let def = defs.get(&value)?;
        match def {
            // assert(cmp .op a b) → assert_cmp .op a b
            OpCode::Cmp {
                kind,
                result: _,
                lhs,
                rhs,
            } => Some(vec![OpCode::AssertCmp {
                kind: *kind,
                lhs: *lhs,
                rhs: *rhs,
            }]),

            // assert(a & b) → assert a; assert b  (for bool AND)
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::And,
                result,
                lhs,
                rhs,
            } => {
                let result_type = function_type_info.get_value_type(*result);
                match result_type.strip_witness().expr {
                    // Only split boolean (u1) AND, not bitwise AND on wider types
                    TypeExpr::U(1) => Some(vec![
                        OpCode::Assert { value: *lhs },
                        OpCode::Assert { value: *rhs },
                    ]),
                    _ => None,
                }
            }

            _ => None,
        }
    }

    /// Try to pull a single-use field multiplication out of a value for R1C formation.
    fn try_pull_mul(
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
