use std::collections::HashSet;

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::Pass,
    ssa::{BlockId, SSA},
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

impl<V: Clone> Pass<V> for RemoveUnreachableBlocks {
    fn run(&self, ssa: &mut SSA<V>, pass_manager: &crate::compiler::pass_manager::PassManager<V>) {
        let cfg = pass_manager.get_cfg();

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

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "remove_unreachable_blocks",
            needs: vec![crate::compiler::pass_manager::DataPoint::CFG],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        true
    }
}
