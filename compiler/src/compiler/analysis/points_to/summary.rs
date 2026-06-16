//! Context-sensitive interprocedural summaries.
//!
//! ## Phase 1 — Polymorphic Summaries
//!
//! Each function is analyzed once with its ref-typed parameters seeded with [`Placeholder`] objects
//! (caller memory the analysis cannot see). The solved points-to relation is projected onto a
//! [`PointsToSummary`]: the objects reaching each return level (a *parametric* transfer function —
//! placeholders stand in for caller arguments), which parameters are written through, and which
//! parameters leak.
//!
//! A call-graph worklist (`compute_summaries`, structurally identical to WTI's) runs this to a
//! least fixpoint, re-queueing callers whenever a callee summary grows. Recursion converges because
//! the object universe is finite and a recursive call's placeholder substitutes into itself as a
//! self-copy the solver drops.
//!
//! When a `Call` is built (see [`super::builder`]), the callee's summary is *instantiated*:
//! returned objects flow into the results with the callee's placeholders substituted by the
//! caller's actual argument objects, written-through arguments are conservatively
//! `External`-polluted, and leaked arguments are escaped. This is the inclusion-valued analog of
//! WTI's `map_formal` summary instantiation.
//!
//! ## Phase 2 — per-context instantiation (`specialize`)
//!
//! A BFS over `(FunctionId, Context)` from `main` re-solves each reachable context with `k ≥ 1`,
//! qualifying each function's local allocations with its call context (`Context::push`) so two call
//! sites of one helper get distinct objects. The builder already re-qualifies callee allocations at
//! instantiation (via the call's `callee_ctx`) and reports the reachable callee contexts
//! (`FunctionConstraints::callee_contexts`), so this phase is just the driving BFS.
//!
//! It terminates because `(FunctionId, Context)` is finite under k-limiting (a recursive call's
//! pushed context repeats and is not re-enqueued).

use std::collections::VecDeque;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            points_to::{
                array_cells::ArrayCells,
                builder::{build_function, ref_levels},
                object::{AbstractObject, Context, NodeKey, Owner, Path},
                solver::PointsToSolution,
            },
            types::TypeInfo,
        },
        ssa::{FunctionId, hlssa::HLSSA},
    },
};

// SUMMARY REPRESENTATION
// ================================================================================================

/// One function's polymorphic points-to summary — its transfer function over formal inputs.
///
/// Object sets are expressed over the function's own universe: its local `Alloc(f, _, empty())`
/// objects, its `Placeholder(f, i, p)` inputs (substituted by the caller's argument objects at a
/// call), `Global`, and `External`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PointsToSummary {
    /// For each return ref-level `(j, path)`, the objects that may reach it.
    pub returns: HashMap<(usize, Path), HashSet<AbstractObject>>,

    /// Precise arg-out.
    ///
    /// For each `(param i, placeholder pointee level `plpath`, cell path `cellpath`)`, the exact
    /// objects the callee stores into that cell of the caller's memory — expressed in the callee's
    /// vocabulary (`Alloc(_,_,empty)`/`Placeholder(_,_,_)`/`Global`/`External`), substituted at the
    /// call. Replaces the old conservative `External`-pollution of written-through params.
    pub param_writes: HashMap<(usize, Path, Path), HashSet<AbstractObject>>,

    /// Parameters whose pointee objects escape inside the function — the caller must treat the
    /// matching argument as escaping.
    pub leaks_param: HashSet<usize>,
}

// PHASE 1: POLYMORPHIC SUMMARY FIXPOINT
// ================================================================================================

/// Compute a polymorphic [`PointsToSummary`] per function via an ascending fixpoint over the call
/// graph (post-order seed from `main`, re-queue callers on summary growth).
pub fn compute_summaries(
    ssa: &HLSSA,
    flow: &FlowAnalysis,
    types: &TypeInfo,
    array_cells: &HashMap<FunctionId, ArrayCells>,
) -> HashMap<FunctionId, PointsToSummary> {
    let main = ssa.get_main_id();
    let fids: Vec<FunctionId> = ssa
        .get_function_ids()
        .filter(|f| types.has_function(*f))
        .collect();

    let mut summaries: HashMap<FunctionId, PointsToSummary> = fids
        .iter()
        .map(|f| (*f, PointsToSummary::default()))
        .collect();

    // Seed callee-first (post-order from main) so summaries are ready before their callers consume
    // them; append any analyzed function unreachable from main so none are dropped.
    let call_graph = flow.get_call_graph();
    let mut order: Vec<FunctionId> = call_graph
        .get_post_order(main)
        .filter(|f| summaries.contains_key(f))
        .collect();
    let mut queued: HashSet<FunctionId> = order.iter().copied().collect();
    for f in &fids {
        if queued.insert(*f) {
            order.push(*f);
        }
    }
    let mut worklist: VecDeque<FunctionId> = order.into();

    while let Some(f) = worklist.pop_front() {
        queued.remove(&f);
        let new = analyze_function(ssa, types, f, &summaries, main, array_cells);
        if summaries[&f] != new {
            summaries.insert(f, new);
            for c in call_graph.get_callers(f) {
                if summaries.contains_key(&c) && queued.insert(c) {
                    worklist.push_back(c);
                }
            }
        }
    }

    summaries
}

/// Analyze one function against the current callee summaries and project out its summary.
fn analyze_function(
    ssa: &HLSSA,
    types: &TypeInfo,
    fid: FunctionId,
    summaries: &HashMap<FunctionId, PointsToSummary>,
    main: FunctionId,
    array_cells: &HashMap<FunctionId, ArrayCells>,
) -> PointsToSummary {
    let func = ssa.get_function(fid);
    let fc = build_function(
        ssa,
        func,
        types.get_function(fid),
        &array_cells[&fid],
        fid,
        &Context::empty(),
        summaries,
        fid == main,
        0,
    );
    let sol = fc.constraints.solve();
    let escaped = super::compute_escaped(&sol, &fc.escape_roots);

    // Returns: the objects reaching each return ref-level.
    let mut returns: HashMap<(usize, Path), HashSet<AbstractObject>> = HashMap::default();
    for (j, ret_ty) in func.get_returns().iter().enumerate() {
        for path in ref_levels(ret_ty) {
            let objs = sol.get(&NodeKey::Val(Owner::Return(j), path.clone()));
            if !objs.is_empty() {
                returns.insert((j, path), objs.clone());
            }
        }
    }

    // A parameter leaks iff its placeholder reaches a sink in the escape closure.
    let mut leaks_param: HashSet<usize> = HashSet::default();
    for o in &escaped {
        if let AbstractObject::Placeholder(f2, i, _) = o {
            if *f2 == fid {
                leaks_param.insert(*i);
            }
        }
    }

    // Precise arg-out: capture the exact objects the callee stored into each placeholder cell (what
    // it wrote into caller memory through a ref parameter), keyed by (param, pointee level, cell).
    let mut param_writes: HashMap<(usize, Path, Path), HashSet<AbstractObject>> =
        HashMap::default();
    for (node, objs) in sol.iter() {
        if let NodeKey::Obj(AbstractObject::Placeholder(f2, i, plpath), cellpath) = node {
            if *f2 == fid && !objs.is_empty() {
                param_writes
                    .entry((*i, plpath.clone(), cellpath.clone()))
                    .or_default()
                    .extend(objs.iter().cloned());
            }
        }
    }

    PointsToSummary {
        returns,
        param_writes,
        leaks_param,
    }
}

// PHASE 2: PER-CONTEXT INSTANTIATION
// ================================================================================================

/// Re-solve every reachable `(function, context)` from `main`, qualifying local allocations with
/// the call context so distinct call sites of one helper produce distinct objects.
///
/// `k` bounds the call-string depth (`1` for 1-CFA). Returns one solution per reachable context;
/// `main` appears under [`Context::empty`].
pub fn specialize(
    ssa: &HLSSA,
    types: &TypeInfo,
    summaries: &HashMap<FunctionId, PointsToSummary>,
    k: usize,
    array_cells: &HashMap<FunctionId, ArrayCells>,
) -> HashMap<(FunctionId, Context), PointsToSolution> {
    let main = ssa.get_main_id();
    let mut results: HashMap<(FunctionId, Context), PointsToSolution> = HashMap::default();
    let mut seen: HashSet<(FunctionId, Context)> = HashSet::default();
    let mut worklist: VecDeque<(FunctionId, Context)> = VecDeque::new();

    let start = (main, Context::empty());
    seen.insert(start.clone());
    worklist.push_back(start);

    while let Some((fid, ctx)) = worklist.pop_front() {
        if !types.has_function(fid) {
            continue;
        }
        let fc = build_function(
            ssa,
            ssa.get_function(fid),
            types.get_function(fid),
            &array_cells[&fid],
            fid,
            &ctx,
            summaries,
            fid == main,
            k,
        );
        let sol = fc.constraints.solve();
        for cc in &fc.callee_contexts {
            if seen.insert(cc.clone()) {
                worklist.push_back(cc.clone());
            }
        }
        results.insert((fid, ctx), sol);
    }

    results
}
