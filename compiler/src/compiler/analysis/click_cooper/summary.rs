//! The interprocedural layer: a return-determinism pass (Phase 0), polymorphic jump-function
//! summaries (Phase 1), a symbolic-congruence projection over the converged summaries, and 1-CFA
//! per-context specialization (Phase 2).
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
            click_cooper::{
                congruence::{Congruence, OpKey, op_signature, pure_op_operands},
                def_order::DefOrder,
                lattice::{Constness, const_join},
                solver::{FunctionFacts, solve_with_writeback},
            },
            flow_analysis::FlowAnalysis,
            shared::{call_string::Context, fixpoint::call_graph_fixpoint},
            value_definitions::{FunctionValueDefinitions, ValueDefinition},
        },
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{CallTarget, Constant, HLFunction, HLSSA, HLSSAConstantsSnapshot, OpCode},
        },
    },
};

// CONSTANTS
// ================================================================================================

/// Call-string depth for the context-sensitive (Phase 2) pass — 1-CFA.
const K: usize = 1;

// DETERMINISM (PHASE 0)
// ================================================================================================

/// Per-function return-position determinism: `det[g][j] == true` iff return position `j` of `g` is
/// a deterministic function of `g`'s arguments.
pub(crate) type DetSummaries = HashMap<FunctionId, Vec<bool>>;

/// Solve every function's per-return determinism to a fixpoint over the call graph.
pub(crate) fn compute_determinism(ssa: &HLSSA, flow: &FlowAnalysis) -> DetSummaries {
    // Optimistic: every return position starts deterministic; the worklist only ever lowers a bit
    // to false. A return is deterministic if its body's non-deterministic sources never reach it,
    // which (for a recursive function) holds at the greatest fixpoint started from "all true".
    let fids: Vec<FunctionId> = ssa.get_function_ids().collect();
    call_graph_fixpoint(
        ssa,
        flow,
        &fids,
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
) -> (HashMap<FunctionId, FnSummary>, SymSolveCache) {
    // Polymorphic over call sites: callees folded via their current summaries, parameters `Bottom`.
    let fids: Vec<FunctionId> = ssa.get_function_ids().collect();

    // The worklist discards each round's `FunctionFacts`, but a function's *last* analysis runs
    // against converged callee summaries (any callee change re-queues it), so those facts are
    // exactly what the symbolic post-pass would recompute. Retain the sym-relevant slice as we go —
    // the `FnMut` closure captures the cache, mirroring `witness_taint_inference`'s graph capture.
    let mut sym_cache = SymSolveCache::default();
    let summaries = call_graph_fixpoint(
        ssa,
        flow,
        &fids,
        |_| FnSummary::default(),
        |f, summaries| {
            let func = ssa.get_function(f);
            if func.get_returns().is_empty() {
                // Arity 0: nothing flows out and the symbolic pass skips it — no solve, no cache.
                return FnSummary::default();
            }
            let facts = solve_polymorphic(func, consts, det, summaries);
            let summary = summarize_returns(ssa, consts, f, &facts);

            // Keep only what `compute_sym_summaries` reads; drop `lattice`/`exec_edges`/`block_facts`.
            let FunctionFacts {
                reachable,
                congruence,
                ..
            } = facts;
            sym_cache.insert(
                f,
                SymSolveFacts {
                    reachable,
                    congruence,
                },
            );
            summary
        },
    );
    (summaries, sym_cache)
}

/// Project `fid`'s return jump functions from its already-solved polymorphic `facts`.
///
/// The polymorphic solve applies the combined-fixpoint writeback (params `Bottom`): a return that is
/// a comparison of congruent operands folds to a constant, so its summary becomes `Const`.
fn summarize_returns(
    ssa: &HLSSA,
    consts: &HLSSAConstantsSnapshot,
    fid: FunctionId,
    facts: &FunctionFacts,
) -> FnSummary {
    let func = ssa.get_function(fid);
    let arity = func.get_returns().len();
    let params: Vec<ValueId> = param_values(ssa, fid);
    let returns = reachable_returns(func, &facts.reachable);

    // No reachable return (the function always traps): nothing flows out.
    if returns.is_empty() {
        return FnSummary {
            returns: vec![ReturnJump::Bottom; arity],
        };
    }

    let jumps = (0..arity)
        .map(|j| return_jump(consts, facts, &params, &returns, j))
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

// SYMBOLIC SUMMARIES
// ================================================================================================

/// Maximum depth of a symbolic return expression.
///
/// Set low deliberately: past shallow arithmetic the binding constraint on expressibility is
/// opacity (witnesses, φ, memory, nested calls), not depth, so a deeper cap only grows the
/// grafted-node count for tree shapes that rarely occur. A `Param` or `Const` leaf is admitted at
/// any depth (it terminates recursion); only descending *into* an op consumes budget.
const MAX_DEPTH: usize = 2;

/// A bounded-depth symbolic expression over a function's formals — the congruence half of an
/// interprocedural return jump function.
///
/// A return value expressed as such a tree can be *grafted* into a caller's value numbering at a
/// call site (substituting the actual arguments for `Param` leaves and the interned constants for
/// `Const` leaves), so the constrained call's result is numbered by the expression rather than left
/// an opaque `CallDet` node. This both ignores arguments the return does not use and relates the
/// result to an open expression over the used ones.
// No `PartialEq`/`Hash`: agreement across a function's returns is decided by the congruence
// partition (`known_equal`), never by structural comparison of `Sym` trees.
#[derive(Clone, Debug)]
pub(crate) enum Sym {
    /// Formal parameter `i`.
    Param(usize),

    /// A program-interned scalar constant, named by its (program-wide) value id.
    Const(ValueId),

    /// A pure value-numbering op (its commutativity flag carried verbatim from [`op_signature`],
    /// the single source of truth) applied to operand sub-expressions.
    Op(OpKey, bool, Vec<Sym>),
}

/// Per-function, per-return-position symbolic congruence jump functions.
///
/// A position holds `Some(sym)` only when `sym` is `Op`-rooted: a bare `Param`/`Const` root is
/// already covered by the constant/pass-through [`ReturnJump`] and has no representation in the
/// split-only congruence partition (there is no "alias `r` to a leaf" primitive).
pub(crate) type SymSummaries = HashMap<FunctionId, Vec<Option<Sym>>>;

/// Project each function's per-return symbolic congruence jump functions over the converged
/// polymorphic summaries.
///
/// This is a **post-fixpoint** pass: it runs after [`compute_summaries`] has converged and does not
/// feed back into it (the symbolic data lives in a separate map, never on [`FnSummary`]), so it is
/// structurally output-only and leaves the summary fixpoint's termination argument untouched.
///
/// It reuses the polymorphic facts [`compute_summaries`] captured in `cache` (the final-round solve
/// per function) rather than re-solving — see [`SymSolveCache`].
pub(crate) fn compute_sym_summaries(
    ssa: &HLSSA,
    consts: &HLSSAConstantsSnapshot,
    cache: &SymSolveCache,
) -> SymSummaries {
    let mut out = SymSummaries::default();
    for fid in ssa.get_function_ids() {
        let func = ssa.get_function(fid);
        let arity = func.get_returns().len();
        if arity == 0 {
            continue;
        }

        // The polymorphic facts (params `Bottom`, callees folded via their summaries, sym channel
        // OFF) captured during `compute_summaries`. Absent only for arity-0 functions, skipped above.
        let Some(facts) = cache.get(&fid) else {
            continue;
        };

        let params = param_values(ssa, fid);
        let defs = FunctionValueDefinitions::from_function(func);
        let returns = reachable_returns(func, &facts.reachable);
        if returns.is_empty() {
            continue;
        }

        let syms: Vec<Option<Sym>> = (0..arity)
            .map(|j| return_sym(&params, &defs, &facts.congruence, consts, &returns, j))
            .collect();

        // Drop the all-`None` case to keep the map (and the grafting fast path) sparse.
        if syms.iter().any(Option::is_some) {
            out.insert(fid, syms);
        }
    }
    out
}

/// The symbolic congruence jump for return position `j`.
///
/// When every reachable return at `j` is present and mutually congruent, this is the first return's
/// expression — valid for all of them, since congruent values are structurally equal in every run.
/// `None` if the returns disagree, some return lacks position `j`, or the first return is not a
/// bounded-depth pure `Op`-rooted function of the formals + interned constants.
fn return_sym(
    params: &[ValueId],
    defs: &FunctionValueDefinitions,
    congruence: &Congruence,
    consts: &HLSSAConstantsSnapshot,
    returns: &[&Vec<ValueId>],
    j: usize,
) -> Option<Sym> {
    // All reachable returns must yield a value at `j` and all must be congruent: they then compute
    // the same value in every run, so the first return's expression is a sound graft for all. The
    // congruence partition already subsumes structural (incl. commutative) equality.
    let first = *returns.first()?.get(j)?;
    for vals in &returns[1..] {
        let rv = *vals.get(j)?;
        if rv != first && !congruence.known_equal(rv, first) {
            return None;
        }
    }

    // Build the common expression once, from the first return. Only an `Op`-rooted root is stored
    // (a bare `Param`/`Const` root is already covered by the constant/pass-through `ReturnJump`).
    let sym = build_sym(first, 0, params, defs, congruence, consts)?;
    matches!(sym, Sym::Op(..)).then_some(sym)
}

/// Build the symbolic expression for value `v` over the formals `params`, or `None` if `v` is not a
/// bounded-depth pure function of the formals and program-interned scalar constants.
///
/// A `Param`/`Const` leaf is admitted at any depth; an op consumes one unit of depth budget to
/// descend, so the deepest *expanded op* sits at `MAX_DEPTH - 1`. A nested call, witness, memory
/// op, or non-entry φ (none congruent to a formal) classifies as `None` (opaque), as does an op
/// past the depth budget.
fn build_sym(
    v: ValueId,
    depth: usize,
    params: &[ValueId],
    defs: &FunctionValueDefinitions,
    congruence: &Congruence,
    consts: &HLSSAConstantsSnapshot,
) -> Option<Sym> {
    // Formal pass-through first (a formal is an entry block parameter, never an instruction). A
    // value congruent to a formal is equal to it in every run, so substituting `Param(i)` is sound.
    for (i, p) in params.iter().enumerate() {
        if v == *p || congruence.known_equal(v, *p) {
            return Some(Sym::Param(i));
        }
    }

    // A program-interned *scalar* constant leaf. Aggregate constants are never surfaced, so an
    // interned aggregate is opaque for symbolic purposes.
    if let Some(c) = consts.get(&v) {
        return c.is_scalar().then_some(Sym::Const(v));
    }

    // Otherwise `v` must be a pure op all of whose operands are themselves expressible.
    match defs.get_definition(v) {
        Some(ValueDefinition::Instruction(_, _, op)) => {
            let (key, operands, commutative) = op_signature(op)?;
            if depth >= MAX_DEPTH {
                return None;
            }
            let children: Option<Vec<Sym>> = operands
                .iter()
                .map(|o| build_sym(*o, depth + 1, params, defs, congruence, consts))
                .collect();
            Some(Sym::Op(key, commutative, children?))
        }

        // A non-entry φ not congruent to a formal, or no structural definition: opaque.
        Some(ValueDefinition::Param(..)) | None => None,
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
    sym: &SymSummaries,
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

        // The context-specialized solve applies the combined-fixpoint writeback over the seeded
        // parameters, so a comparison of operands congruent in this context folds in its facts. The
        // symbolic channel is on here, so a call's grafted return expression can refine per-context
        // congruence too.
        let func = ssa.get_function(f);
        let mut facts =
            solve_with_writeback(func, consts, det, Some(summaries), Some(sym), &seed_map);

        // Build the dominance-aware congruence leaders so `leader_in` yields a legal redirect
        // target.
        let order = DefOrder::new(func, flow.get_function_cfg(f));
        facts.congruence.compute_leaders(&order);

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

// FACTS AND FACT CACHE
// ================================================================================================

/// The slice of a function's polymorphic [`FunctionFacts`] that the symbolic post-pass reads,
/// retained from the summary fixpoint's final round so [`compute_sym_summaries`] need not re-solve.
///
/// Only the two fields [`return_sym`]/[`reachable_returns`] consume are kept;
/// `lattice`/`exec_edges`/`block_facts` are dropped so the retained footprint stays minimal.
pub(crate) struct SymSolveFacts {
    pub reachable: HashSet<BlockId>,
    pub congruence: Congruence,
}

/// The per-function [`SymSolveFacts`] captured by [`compute_summaries`] for
/// [`compute_sym_summaries`].
pub(crate) type SymSolveCache = HashMap<FunctionId, SymSolveFacts>;

// HELPERS
// ================================================================================================

/// The polymorphic solve driving the summary fixpoint (the `analyze` closure in
/// [`compute_summaries`]); its final-round facts are cached for [`compute_sym_summaries`].
///
/// Parameters left `Bottom`, callees folded via their current `summaries`, and the symbolic
/// congruence channel off (this is where `Sym` is *produced*, so it must never be consumed here).
fn solve_polymorphic(
    func: &HLFunction,
    consts: &HLSSAConstantsSnapshot,
    det: &DetSummaries,
    summaries: &HashMap<FunctionId, FnSummary>,
) -> FunctionFacts {
    solve_with_writeback(
        func,
        consts,
        det,
        Some(summaries),
        None,
        &HashMap::default(),
    )
}

/// The value lists of every reachable `Return` in `func`, in block order.
fn reachable_returns<'a>(
    func: &'a HLFunction,
    reachable: &HashSet<BlockId>,
) -> Vec<&'a Vec<ValueId>> {
    func.get_blocks()
        .filter(|(bid, _)| reachable.contains(*bid))
        .filter_map(|(_, block)| match block.get_terminator() {
            Some(Terminator::Return(vals)) => Some(vals),
            _ => None,
        })
        .collect()
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
