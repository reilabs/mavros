//! This pass is intended to be the first step of the witness generation phase, and converts every
//! witness write into a side-effect-only sink, with witnesses now flowing as plain values in the
//! CFG.
//!
//! Witness-representation casts (`ValueOf`, `WitnessOf`, and `Map`s thereof at any depth) are NOT
//! handled here: the `StripWitnessOf` pass that follows aliases all of them away.

use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::hlssa::{HLSSA, OpCode},
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
                        OpCode::Guard { inner, .. } => {
                            if let OpCode::WriteWitness {
                                result: r,
                                value: b,
                                ..
                            } = inner.as_mut()
                            {
                                if let Some(r) = r {
                                    replacements.insert(*r, *b);
                                }
                                *r = None;
                            }
                        }
                        _ => {}
                    }
                }
            }

            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    replacements.replace_instruction(instruction);
                }
                replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }
}
