use std::collections::HashSet;

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::SSA,
};

/// Removes functions that are not reachable from main via the call graph.
/// This pass should run after defunctionalization, when all calls are static.
pub struct RemoveUnreachableFunctions {}

impl RemoveUnreachableFunctions {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for RemoveUnreachableFunctions {
    fn name(&self) -> &'static str { "remove_unreachable_functions" }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut SSA, store: &AnalysisStore) {
        let cfg = store.get::<FlowAnalysis>();
        let call_graph = cfg.get_call_graph();

        let main_id = ssa.get_main_id();
        let reachable: HashSet<_> = call_graph.get_post_order(main_id).collect();

        let all_function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in all_function_ids {
            if !reachable.contains(&function_id) {
                ssa.take_function(function_id);
            }
        }
    }
}
