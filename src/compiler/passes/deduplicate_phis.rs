use std::collections::HashMap;

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{BlockId, SSA, Terminator, ValueId},
};

pub struct DeduplicatePhis {}

impl Pass for DeduplicatePhis {
    fn name(&self) -> &'static str {
        "deduplicate_phis"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut SSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
}

impl DeduplicatePhis {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut SSA) {
        for (_, function) in ssa.iter_functions_mut() {
            let mut unifications: HashMap<(BlockId, Vec<ValueId>), Vec<BlockId>> = HashMap::new();
            for (block_id, block) in function.get_blocks() {
                match block.get_terminator() {
                    Some(Terminator::Jmp(target, inputs)) => {
                        if inputs.len() > 0 {
                            unifications
                                .entry((*target, inputs.clone()))
                                .or_insert(vec![])
                                .push(*block_id);
                        }
                    }
                    _ => {}
                }
            }

            for ((target, inputs), blocks) in unifications {
                if blocks.len() <= 1 {
                    continue;
                }
                let new_block = function.add_block();
                function.terminate_block_with_jmp(new_block, target, inputs);
                for block_id in blocks {
                    function.terminate_block_with_jmp(block_id, new_block, vec![]);
                }
            }
        }
    }
}
