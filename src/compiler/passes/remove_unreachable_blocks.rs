use std::collections::HashSet;

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    ssa::{BlockId, HLSSA},
};

// Needs to happen because apparently Noir
// produces dead code paths that have no predecessors.
// TODO: Check if we need this with our own SSA gen.
pub struct RemoveUnreachableBlocks {}

impl RemoveUnreachableBlocks {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for RemoveUnreachableBlocks {
    fn name(&self) -> &'static str {
        "remove_unreachable_blocks"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let cfg = store.get::<FlowAnalysis>();

        for (function_id, function) in ssa.iter_functions_mut() {
            let function_cfg = cfg.get_function_cfg(*function_id);
            let reachable: HashSet<BlockId> = function_cfg.get_domination_pre_order().collect();
            let all_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
            for block_id in all_blocks {
                if !reachable.contains(&block_id) {
                    _ = function.take_block(block_id);
                }
            }
        }
    }
}
