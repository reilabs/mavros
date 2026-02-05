use std::collections::HashSet;

use crate::compiler::{
    pass_manager::{DataPoint, Pass, PassInfo, PassManager},
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

impl<V: Clone> Pass<V> for RemoveUnreachableFunctions {
    fn run(&self, ssa: &mut SSA<V>, pass_manager: &PassManager<V>) {
        let cfg = pass_manager.get_cfg();
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

    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "remove_unreachable_functions",
            needs: vec![DataPoint::CFG],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        true
    }
}
