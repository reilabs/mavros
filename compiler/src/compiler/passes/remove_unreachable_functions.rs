//! Removes functions that are not reachable from main via the call graph.
//!
//! This pass should run after defunctionalization, and entry point preparation. This ensures that
//! all calls are static and hence avoids producing incorrect results.

use crate::{
    collections::HashSet,
    compiler::{
        analysis::flow_analysis::FlowAnalysis,
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        ssa::hlssa::HLSSA,
    },
};

pub struct RemoveUnreachableFunctions {}

impl RemoveUnreachableFunctions {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for RemoveUnreachableFunctions {
    fn name(&self) -> &'static str {
        "remove_unreachable_functions"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let cfg = store.get::<FlowAnalysis>();
        let call_graph = cfg.get_call_graph();

        let main_id = ssa.get_main_id();
        let reachable: HashSet<_> = call_graph.get_post_order(main_id).collect();

        let all_function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in all_function_ids {
            if !reachable.contains(&function_id) {
                // If we get here then the function is not reachable so there will be no dangling
                // references to it.
                ssa.delete_function(function_id);
            }
        }
    }
}
