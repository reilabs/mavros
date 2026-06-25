//! The interprocedural layer: a return-determinism pass (Phase 0), polymorphic jump-function
//! summaries (Phase 1) and 1-CFA per-context specialization (Phase 2), mirroring the two-phase
//! structure of `analysis/points_to`.
//!
//! # Phase 0: Return Determinism
//!
//! [`compute_determinism`] records, per function and per return position, whether that return is a
//! *deterministic function of the function's arguments*.
//!
//! It is a greatest-fixpoint: every return starts assumed deterministic and a return is lowered to
//! non-deterministic when its value's backward slice reaches a non-deterministic source (a fresh
//! witness, a memory read, a global, an unconstrained or dynamic call, …) or a non-deterministic
//! callee return. These bits drive the *cross-call congruence* in [`super::congruence`]: two
//! constrained static calls to one callee with operandwise-congruent arguments produce congruent
//! results exactly when the callee's return is deterministic.
//!
//! # Phase 1: Summaries
//!
//! [`compute_summaries`] runs a call-graph worklist and, for each function, records a
//! [`ReturnJump`] per return position. The return is a `Const`, a pass-through of a formal
//! parameter (`Param(i)`), or `Bottom`.
//!
//! A function is solved with its callees' current summaries (so a return that forwards a
//! constant-returning call is itself `Const`), but with parameters left `Bottom` — the summary is
//! *polymorphic* over call sites. The summary lattice is finite, so the worklist converges.
//!
//! # Phase 2: Specialization
//!
//! [`specialize`] re-solves each reachable `(function, context)` with its entry parameters seeded
//! by the calling context's argument constants, so two call sites of one helper get distinct
//! per-context constants. A context is a `k`-limited call string ([`K`]).
//!
//! This is sound without an address-token exclusion as a context's parameter seeds are the **meet**
//! of the argument constants over _every_ static call path that reaches it. Thus, a per-context
//! constant holds on every concrete invocation mapped to that context. Seeds are monotone and
//! contexts finite, so the specialization worklist converges.

use std::{collections::VecDeque, sync::Arc};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            call_string::Context,
            click_cooper::{
                congruence::pure_op_operands,
                lattice::{Constness, const_join},
                solver::{FunctionFacts, FunctionSolver},
            },
            flow_analysis::FlowAnalysis,
        },
        ssa::{
            FunctionId, Instruction, Terminator, ValueId,
            hlssa::{CallTarget, Constant, HLSSA, HLSSAConstantsSnapshot, OpCode},
        },
    },
};

// CONSTANTS
// ================================================================================================

/// Call-string depth for the context-sensitive (Phase 2) pass — 1-CFA.
const K: usize = 1;

// DETERMINISM (PHASE 0)
// ================================================================================================

/// Per-function return-position determinism: `det[g][j] == true` iff return position `j` of `g` is a
/// deterministic function of `g`'s arguments.
pub(crate) type DetSummaries = HashMap<FunctionId, Vec<bool>>;

/// Solve every function's per-return determinism to a fixpoint over the call graph.
pub(crate) fn compute_determinism(ssa: &HLSSA, flow: &FlowAnalysis) -> DetSummaries {
    // Optimistic: every return position starts deterministic; the worklist only ever lowers a bit to
    // false. A return is deterministic if its body's non-deterministic sources never reach it, which
    // (for a recursive function) holds at the greatest fixpoint started from "all true".
    call_graph_fixpoint(
        ssa,
        flow,
        |f| vec![true; ssa.get_function(f).get_returns().len()],
        |f, det| analyze_determinism(ssa, det, f),
    )
}

/// Project `fid`'s per-return determinism given its callees' current bits.
///
/// A value is **non-deterministic** ("tainted") if produced by a non-deterministic source, by a
/// pure operator/φ with a tainted operand, or by a constrained static call whose corresponding
/// return is non-deterministic (or which is passed a tainted argument). Taint is propagated forward
/// to a fixpoint, *ignoring reachability* (a value tainted only on a dead edge is conservatively
/// kept tainted, which is sound). Return `j` is deterministic iff no `Return`'s `j`-th value is
/// tainted.
fn analyze_determinism(ssa: &HLSSA, det: &DetSummaries, fid: FunctionId) -> Vec<bool> {
    let func = ssa.get_function(fid);
    let arity = func.get_returns().len();
    if arity == 0 {
        return Vec::new();
    }

    let mut tainted: HashSet<ValueId> = HashSet::default();
    loop {
        let mut changed = false;
        for (_, block) in func.get_blocks() {
            for instr in block.get_instructions() {
                let results: Vec<ValueId> = instr.get_results().copied().collect();
                if results.is_empty() {
                    continue;
                }
                let taints: Vec<bool> = if let Some(operands) = pure_op_operands(instr) {
                    // A pure value-numbering op (single result): tainted iff some operand is.
                    let t = operands.iter().any(|o| tainted.contains(o));
                    vec![t; results.len()]
                } else if let OpCode::Call {
                    function: CallTarget::Static(g),
                    args,
                    unconstrained: false,
                    ..
                } = instr
                {
                    // A constrained static call: result `j` is deterministic only if the callee's
                    // return `j` is and every argument is.
                    let arg_tainted = args.iter().any(|a| tainted.contains(a));
                    (0..results.len())
                        .map(|j| {
                            arg_tainted || det.get(g).and_then(|r| r.get(j)).copied() != Some(true)
                        })
                        .collect()
                } else {
                    // A non-deterministic source: witnesses, memory, globals, lookups, dynamic or
                    // unconstrained calls, and anything else not a deterministic function of its
                    // operands.
                    vec![true; results.len()]
                };
                for (r, t) in results.iter().zip(taints) {
                    if t && tainted.insert(*r) {
                        changed = true;
                    }
                }
            }

            // A φ (block parameter) is tainted if any incoming jump-arg at its index is tainted.
            if let Some(Terminator::Jmp(target, args)) = block.get_terminator() {
                let params: Vec<ValueId> = func
                    .get_block(*target)
                    .get_parameter_values()
                    .copied()
                    .collect();
                for (i, a) in args.iter().enumerate() {
                    if i < params.len() && tainted.contains(a) && tainted.insert(params[i]) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut det_bits = vec![true; arity];
    for (_, block) in func.get_blocks() {
        if let Some(Terminator::Return(vals)) = block.get_terminator() {
            for (j, v) in vals.iter().enumerate() {
                if j < arity && tainted.contains(v) {
                    det_bits[j] = false;
                }
            }
        }
    }
    det_bits
}

// SUMMARIES (PHASE 1)
// ================================================================================================

/// How one return value relates to a function's formals — a constant-propagation jump function.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ReturnJump {
    /// Always this constant, regardless of arguments.
    Const(Arc<Constant>),

    /// Equal to formal parameter `i` (a pass-through): at a call site the result takes argument
    /// `i`'s constant.
    ///
    /// Also a congruence pass-through (`result ≡ arg i`).
    Param(usize),

    /// Unknown.
    Bottom,
}

/// A function's interprocedural summary: a jump function per return position.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct FnSummary {
    pub returns: Vec<ReturnJump>,
}

/// Phase 1: solve every function's polymorphic summary to a fixpoint over the call graph.
pub(crate) fn compute_summaries(
    ssa: &HLSSA,
    flow: &FlowAnalysis,
    consts: &HLSSAConstantsSnapshot,
    det: &DetSummaries,
) -> HashMap<FunctionId, FnSummary> {
    // Polymorphic over call sites: callees folded via their current summaries, parameters `Bottom`.
    call_graph_fixpoint(
        ssa,
        flow,
        |_| FnSummary::default(),
        |f, summaries| analyze_function(ssa, consts, det, f, summaries),
    )
}

/// Solve `fid` polymorphically (parameters `Bottom`, callees folded via `summaries`) and project
/// its return jump functions.
fn analyze_function(
    ssa: &HLSSA,
    consts: &HLSSAConstantsSnapshot,
    det: &DetSummaries,
    fid: FunctionId,
    summaries: &HashMap<FunctionId, FnSummary>,
) -> FnSummary {
    let func = ssa.get_function(fid);
    let arity = func.get_returns().len();
    if arity == 0 {
        return FnSummary::default();
    }

    let mut solver = FunctionSolver::new(func, consts)
        .with_summaries(summaries)
        .with_determinism(det);
    solver.run();
    let facts = solver.into_facts();

    let params: Vec<ValueId> = param_values(ssa, fid);
    let returns: Vec<&Vec<ValueId>> = func
        .get_blocks()
        .filter(|(bid, _)| facts.reachable.contains(*bid))
        .filter_map(|(_, block)| match block.get_terminator() {
            Some(Terminator::Return(vals)) => Some(vals),
            _ => None,
        })
        .collect();

    // No reachable return (the function always traps): nothing flows out.
    if returns.is_empty() {
        return FnSummary {
            returns: vec![ReturnJump::Bottom; arity],
        };
    }

    let jumps = (0..arity)
        .map(|j| return_jump(consts, &facts, &params, &returns, j))
        .collect();
    FnSummary { returns: jumps }
}

/// The jump function for return position `j`, joined over every reachable `Return`.
fn return_jump(
    consts: &HLSSAConstantsSnapshot,
    facts: &FunctionFacts,
    params: &[ValueId],
    returns: &[&Vec<ValueId>],
    j: usize,
) -> ReturnJump {
    // Constant only if every reachable return yields the same constant at position `j`.
    let mut c = Constness::Top;
    let mut all_present = true;
    for vals in returns {
        match vals.get(j) {
            Some(rv) => c = const_join(c, value_const(consts, facts, *rv)),
            None => all_present = false,
        }
    }
    if let Constness::Const(k) = c {
        return ReturnJump::Const(k);
    }
    if !all_present {
        return ReturnJump::Bottom;
    }

    // Otherwise a pass-through if some single parameter is congruent to the return on *every* path.
    // One scratch set, refined monotonically to the running intersection: later returns only
    // re-check the shrinking candidate set, and an empty intersection short-circuits the rest.
    let mut candidates: Option<HashSet<usize>> = None;
    for vals in returns {
        let rv = vals[j];
        match &mut candidates {
            None => {
                candidates = Some(
                    (0..params.len())
                        .filter(|&i| rv == params[i] || facts.congruence.known_equal(rv, params[i]))
                        .collect(),
                );
            }
            Some(set) => {
                set.retain(|&i| rv == params[i] || facts.congruence.known_equal(rv, params[i]));
                if set.is_empty() {
                    break;
                }
            }
        }
    }
    match candidates.and_then(|set| set.iter().min().copied()) {
        Some(i) => ReturnJump::Param(i),
        None => ReturnJump::Bottom,
    }
}

// SPECIALIZATION (PHASE 2)
// ================================================================================================

/// Phase 2: re-solve every reachable `(function, context)` with parameters seeded by the calling
/// context's argument constants.
pub(crate) fn specialize(
    ssa: &HLSSA,
    flow: &FlowAnalysis,
    consts: &HLSSAConstantsSnapshot,
    summaries: &HashMap<FunctionId, FnSummary>,
    det: &DetSummaries,
) -> HashMap<(FunctionId, Context), FunctionFacts> {
    let main = ssa.get_unique_entrypoint_id();

    // Entry-parameter values per function, collected once and borrowed throughout: the worklist
    // would otherwise re-walk a function's parameters on every dequeue and every visited call site.
    let param_index: HashMap<FunctionId, Vec<ValueId>> = ssa
        .get_function_ids()
        .map(|f| (f, param_values(ssa, f)))
        .collect();

    // Per-context entry-parameter seeds, met over all reaching static call paths. A `Top` seed
    // means "no path seen yet"; joining the first argument constant replaces it.
    let mut seeds: HashMap<(FunctionId, Context), Vec<Constness>> = HashMap::default();
    let mut results: HashMap<(FunctionId, Context), FunctionFacts> = HashMap::default();
    let mut in_worklist: HashSet<(FunctionId, Context)> = HashSet::default();
    let mut worklist: VecDeque<(FunctionId, Context)> = VecDeque::new();

    // `main`'s arguments are external inputs: `Bottom`.
    let start = (main, Context::empty());
    seeds.insert(
        start.clone(),
        vec![Constness::Bottom; param_index[&main].len()],
    );
    in_worklist.insert(start.clone());
    worklist.push_back(start);

    while let Some((f, ctx)) = worklist.pop_front() {
        in_worklist.remove(&(f, ctx.clone()));

        let params = &param_index[&f];
        let seed_vec = seeds
            .get(&(f, ctx.clone()))
            .cloned()
            .unwrap_or_else(|| vec![Constness::Bottom; params.len()]);
        let seed_map: HashMap<ValueId, Constness> = params
            .iter()
            .copied()
            .zip(seed_vec.into_iter().map(seed_or_bottom))
            .collect();

        let func = ssa.get_function(f);
        let mut solver = FunctionSolver::new(func, consts)
            .with_summaries(summaries)
            .with_determinism(det)
            .with_param_seeds(seed_map);
        solver.run();
        let mut facts = solver.into_facts();

        // Build the dominance-aware congruence leaders so `leader_in` yields a legal redirect
        // target.
        facts
            .congruence
            .compute_leaders(func, flow.get_function_cfg(f));

        // Propagate argument constants from each reachable static call into the callee's context.
        for (bid, block) in func.get_blocks() {
            if !facts.reachable.contains(bid) {
                continue;
            }

            for instr in block.get_instructions() {
                let OpCode::Call {
                    results: call_results,
                    function: CallTarget::Static(g),
                    args,
                    unconstrained: false,
                } = instr
                else {
                    continue;
                };

                let g = *g;
                if !summaries.contains_key(&g) {
                    continue;
                }

                let site = call_results.first().or_else(|| args.first()).copied();
                let g_ctx = match site {
                    Some(v) => ctx.push((f, v), K),
                    None => ctx.clone(),
                };
                let g_params = &param_index[&g];

                let entry = seeds
                    .entry((g, g_ctx.clone()))
                    .or_insert_with(|| vec![Constness::Top; g_params.len()]);
                let mut grew = false;
                for i in 0..g_params.len() {
                    let arg = match args.get(i) {
                        Some(a) => value_const(consts, &facts, *a),
                        None => Constness::Bottom,
                    };
                    let joined = const_join(entry[i].clone(), arg);
                    if joined != entry[i] {
                        entry[i] = joined;
                        grew = true;
                    }
                }

                let unseen = !results.contains_key(&(g, g_ctx.clone()));
                if (grew || unseen) && in_worklist.insert((g, g_ctx.clone())) {
                    worklist.push_back((g, g_ctx));
                }
            }
        }

        results.insert((f, ctx), facts);
    }

    results
}

// HELPERS
// ================================================================================================

/// Solve a per-function fact to a fixpoint over the call graph.
///
/// Every function starts at `init(f)`. The worklist is seeded callee-first (post-order from `main`,
/// then any function unreachable from `main` so none is dropped) and `analyze(f, &map)` re-runs
/// whenever a callee's fact changed, until nothing moves. The accumulator is passed to `analyze`, so
/// a function is solved against its callees' current facts.
fn call_graph_fixpoint<T: PartialEq>(
    ssa: &HLSSA,
    flow: &FlowAnalysis,
    init: impl Fn(FunctionId) -> T,
    analyze: impl Fn(FunctionId, &HashMap<FunctionId, T>) -> T,
) -> HashMap<FunctionId, T> {
    let main = ssa.get_unique_entrypoint_id();
    let fids: Vec<FunctionId> = ssa.get_function_ids().collect();
    let mut map: HashMap<FunctionId, T> = fids.iter().map(|f| (*f, init(*f))).collect();

    let call_graph = flow.get_call_graph();
    let mut order: Vec<FunctionId> = call_graph
        .get_post_order(main)
        .filter(|f| map.contains_key(f))
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

/// The entry-parameter values of `f`, in order.
fn param_values(ssa: &HLSSA, f: FunctionId) -> Vec<ValueId> {
    ssa.get_function(f)
        .get_entry()
        .get_parameters()
        .map(|(p, _)| *p)
        .collect()
}

/// The unconditional constant of `v` in a solved function: an interned constant, the lattice's
/// constant, or `Bottom` (a reachable, used value is never left at `Top`).
fn value_const(consts: &HLSSAConstantsSnapshot, facts: &FunctionFacts, v: ValueId) -> Constness {
    if let Some(c) = consts.get(&v) {
        return Constness::Const(c.clone());
    }
    facts.lattice.get(&v).cloned().unwrap_or(Constness::Bottom)
}

/// A `Top` seed (no reaching path recorded) means the parameter is effectively unconstrained for
/// the solve, so it is started at `Bottom`.
fn seed_or_bottom(c: Constness) -> Constness {
    match c {
        Constness::Top => Constness::Bottom,
        other => other,
    }
}
