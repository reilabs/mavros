use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::{Analysis as _, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{HLSSA, OpCode},
};

pub struct WitnessWriteToVoid {}

impl Pass for WitnessWriteToVoid {
    fn name(&self) -> &'static str {
        "witness_write_to_void"
    }
    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl WitnessWriteToVoid {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut HLSSA) {
        for (_, function) in ssa.iter_functions_mut() {
            let mut replacements = ValueReplacements::new();

            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    match instruction {
                        OpCode::WriteWitness {
                            result: r,
                            value: b,
                            ..
                        } => {
                            if let Some(r) = r {
                                replacements.insert(*r, *b);
                            }
                            *r = None;
                        }
                        OpCode::ValueOf {
                            result: r,
                            value: v,
                        } => {
                            replacements.insert(*r, *v);
                        }
                        OpCode::Guard { inner, .. } => match inner.as_mut() {
                            OpCode::WriteWitness {
                                result: r,
                                value: b,
                                ..
                            } => {
                                if let Some(r) = r {
                                    replacements.insert(*r, *b);
                                }
                                *r = None;
                            }
                            OpCode::ValueOf {
                                result: r,
                                value: v,
                            } => {
                                replacements.insert(*r, *v);
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }

            for (_, block) in function.get_blocks_mut() {
                // Remove ValueOf instructions and Guard-wrapped ValueOf (identity in witgen pipeline)
                let old_instructions = block.take_instructions();
                let new_instructions = old_instructions
                    .into_iter()
                    .filter(|instr| {
                        !matches!(instr, OpCode::ValueOf { .. })
                            && !matches!(instr, OpCode::Guard { inner, .. } if matches!(inner.as_ref(), OpCode::ValueOf { .. }))
                    })
                    .collect();
                block.put_instructions(new_instructions);

                for instruction in block.get_instructions_mut() {
                    replacements.replace_instruction(instruction);
                }
                replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }
}
