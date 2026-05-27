//! Removes redundant traffic from phi inputs at join points, reducing value flow from N edges to 1.

use std::collections::HashMap;

use crate::compiler::{
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    ssa::{BlockId, Terminator, ValueId, hlssa::HLSSA},
};

pub struct DeduplicatePhis {}

impl Pass for DeduplicatePhis {
    fn name(&self) -> &'static str {
        "deduplicate_phis"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![]
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
}

impl DeduplicatePhis {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA) {
        for (_, function) in ssa.iter_functions_mut() {
            let mut unifications: HashMap<(BlockId, Vec<ValueId>), Vec<BlockId>> = HashMap::new();
            for (block_id, block) in function.get_blocks() {
                if let Some(Terminator::Jmp(target, inputs)) = block.get_terminator() {
                    if inputs.len() > 0 {
                        unifications
                            .entry((*target, inputs.clone()))
                            .or_insert(vec![])
                            .push(*block_id);
                    }
                }
            }

            // The iteration order is non-deterministic, but given block IDs shouldn't leak into
            // anything user-facing this isn't really an issue.
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
