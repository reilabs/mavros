use std::collections::HashMap;

use crate::compiler::{
    pass_manager::Pass,
    ssa::{BinaryArithOpKind, OpCode, SSA, ValueId},
};

pub struct PullIntoAssert {}

pub struct PulledProduct {
    lhs: ValueId,
    rhs: ValueId,
}

impl Pass for PullIntoAssert {
    fn run(&self, ssa: &mut SSA, _pass_manager: &crate::compiler::pass_manager::PassManager) {
        self.do_run(ssa);
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "pull_into_assert",
            needs: vec![],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }

    fn invalidates_types(&self) -> bool {
        false
    }
}

impl PullIntoAssert {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut SSA) {
        for (_, function) in ssa.iter_functions_mut() {
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
                            let mut pull = self.try_pull(lhs, &uses, &defs);
                            let mut other_op = rhs;
                            if pull.is_none() {
                                pull = self.try_pull(rhs, &uses, &defs);
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
                result: _,
                lhs,
                rhs,
            } => Some(PulledProduct {
                lhs: *lhs,
                rhs: *rhs,
            }),
            _ => None,
        }
    }
}
