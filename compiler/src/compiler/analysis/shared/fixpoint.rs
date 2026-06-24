//! The shared call-graph fixpoint driver for the interprocedural analyses.
//!
//! The worklist operates as follows:
//!
//! 1. Seed callee-first using a post-order traversal from `main`.
//! 2. Re-analyze each function against its callees' current facts, and re-queue its callers
//!    whenever a fact changes.

use std::collections::VecDeque;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{analysis::flow_analysis::FlowAnalysis, ssa::FunctionId, ssa::hlssa::HLSSA},
};

/// Solve a per-function fact to a fixpoint over the call graph.
///
/// `fids` is the analysis domain of functions to solve. Every function in `fids` starts at
/// `init(f)`. The worklist is seeded callee-first (post-order from `main`, then any in-domain
/// function unreachable from `main` so none is dropped) and `analyze(f, &map)` re-runs whenever a
/// callee's fact changed, until nothing moves. The accumulator is passed to `analyze`, so a
/// function is solved against its callees' current facts.
///
/// `analyze` is `FnMut` so a caller may record per-function side outputs (e.g. the graph built on
/// each pop) as it goes; every function in `fids` is analyzed at least once.
///
/// Termination of this function depends on `analyze` providing a well-formed reduction of the state
/// space.
pub(crate) fn call_graph_fixpoint<T: PartialEq>(
    ssa: &HLSSA,
    flow: &FlowAnalysis,
    fids: &[FunctionId],
    init: impl Fn(FunctionId) -> T,
    mut analyze: impl FnMut(FunctionId, &HashMap<FunctionId, T>) -> T,
) -> HashMap<FunctionId, T> {
    // We start at `main` as our entry point and build the call graph from it.
    let main = ssa.get_unique_entrypoint_id();
    let mut map: HashMap<FunctionId, T> = fids.iter().map(|f| (*f, init(*f))).collect();
    let call_graph = flow.get_call_graph();
    let mut order: Vec<FunctionId> = call_graph
        .get_post_order(main)
        .filter(|f| map.contains_key(f))
        .collect();

    // The initial state of the traversal order is used to seed the queue of functions, which forms
    // the basis of our worklist.
    let mut queued: HashSet<FunctionId> = order.iter().copied().collect();
    for f in fids {
        if queued.insert(*f) {
            order.push(*f);
        }
    }
    let mut worklist: VecDeque<FunctionId> = order.into();

    // Then the worklist loop runs until there is nothing left to do.
    while let Some(f) = worklist.pop_front() {
        queued.remove(&f);
        let new = analyze(f, &map);
        if map[&f] != new {
            map.insert(f, new);
            for c in call_graph.get_callers(f) {
                if map.contains_key(&c) && queued.insert(c) {
                    worklist.push_back(c);
                }
            }
        }
    }

    map
}
