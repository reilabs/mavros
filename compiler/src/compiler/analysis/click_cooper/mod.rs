//! A Click–Cooper-style combined optimistic analysis.
//!
//! An analysis that handles **constants**, **reachability** and **congruence** (using optimistic
//! AWZ value numbering), plus a context-sensitive **interprocedural** (1-CFA) layer. It is designed
//! to drive constant/condition propagation and PRE. The combination is _at least_ as precise as
//! running the factors separately and alternating to a fixpoint (Click & Cooper, _Combining
//! Analyses, Combining Optimizations_, TOPLAS 1995).
//!
//! It operates over three kinds of facts:
//!
//! - **Intraprocedural facts** are those within a given function and are derived from a
//!   Wegman-Zadeck constants and reachability fixpoint with a structural-congruence partition over
//!   the reachability state.
//! - **Interprocedural facts** refine the intraprocedural view across calls: polymorphic
//!   jump-function summaries (a `return_const` holding for any arguments, a `return Param(i)`
//!   pass-through) and 1-CFA per-`(function, context)` specialization seeding callee parameters
//!   with the caller's argument constants, plus _cross-call congruence_: a constrained static call's
//!   result is numbered by grafting the callee's symbolic return expression over the actual
//!   arguments (relating it to an open expression, and ignoring arguments the return does not use),
//!   or — failing that — by a determinism-gated whole-call key (two calls with congruent arguments
//!   yield congruent results).
//! - **Conditional facts** are ones derived from asserts, equalities, disequalities, and unpinned
//!   witness forwarding. They are computed post-convergence over the same reachability state plus
//!   CFG dominance — once from the intraprocedural facts and once per `(function, context)` from
//!   the 1-CFA-specialized facts — and are intentionally disjoint from the unconditional view.
//!
//! **Unconditional facts** are those which hold on any path so that any use (replacing a use,
//! deleting a pure definition, or pruning an unreachable block) maintains soundness. **Conditional
//! facts** are those established through control flow (branches, assertions, witness operations)
//! that are only correct to use in contexts where the establishing constraints are preserved.
//!
//! # Combined-Fixpoint Writeback
//!
//! The constant and reachability factors feed congruence (constants seed the partition,
//! reachability scopes φ-operands), and congruence feeds _back_: a comparison whose operands are
//! proven congruent is a constant the lattice alone cannot derive — `x == y` is unconditionally
//! `true` and `x < y` unconditionally `false` when `x` and `y` are congruent.
//!
//! Each solve therefore alternates: solve constants/reachability, build congruence, promote those
//! comparison results to constants, and re-solve so the new constants fold branches and cascade (a
//! folded branch prunes an edge, which re-scopes φ-operands, which can expose further congruences
//! and constants). This repeats until no new fact appears. It is monotone — promotions only grow,
//! executable edges only shrink — so it converges, and it is strictly stronger than running the
//! constant and congruence factors separately. This is what makes the analysis "combined" rather
//! than merely "staged".
//!
//! # Correctness
//!
//! This analysis is **sound** because each fact class is sound, with the reasoning given as
//! follows:
//!
//! - **Constants:** The transfer functions fold a value only when the result is exact for the
//!   operand widths so `value == c` holds under any advice.
//! - **Aggregate Folding:** The pure, value-semantic sequence ops (`MkSeq`, `MkRepeated`,
//!   `MkSeqOfBlob`, `ArrayGet`, `ArraySet`, `SlicePush`, `SliceLen`) are folded over the same
//!   exact-constant lattice — a projection at an out-of-bounds constant index is refused rather
//!   than guessed — so every folded element equals its runtime value. The aggregate (`Blob`)
//!   constants are never surfaced (only the scalar projections flow out), so a consumer only ever
//!   sees an ordinary scalar constant and inherits the **Constants** soundness above.
//! - **Reachability:** An edge is executable only when a predecessor's terminator can take it. A
//!   block proven unreachable is thus taken in no run.
//! - **Congruence:** Two values are congruent iff computed by the same operator from operand-wise
//!   congruent inputs, hence equal in all runs. Congruence never asserts an equality an adversarial
//!   witness could validate.
//! - **Combined-Fixpoint Writeback:** A comparison of congruent operands is folded to a constant
//!   only via `known_equal`, which holds in every run under any advice (two free witnesses are
//!   never made congruent), so `x == y → true` / `x < y → false` are exact. The resulting edge
//!   folding reuses the reachability mechanism above; only the (pure, scalar) comparison result is
//!   promoted, so the fact stays safe to replace, delete, or prune on.
//! - **Interprocedural Constants:** A `return_const` jump function holds for any arguments; a
//!   `return Param(i)` pass-through equals argument `i` exactly. Per-context parameter seeds are
//!   the meet of the argument constants over _every_ static call path mapped to that context, so a
//!   per-context constant holds on every concrete invocation in that context.
//! - **Cross-Call Congruence:** A constrained static call's result is value-numbered either by
//!   grafting the callee's symbolic return — a bounded-depth expression whose leaves are the
//!   callee's formals (instantiated to the actual arguments) and program constants and whose nodes
//!   are pure, deterministic ops — or, when no such expression exists, by a `(callee, return-index)`
//!   key over its argument classes, gated on the return being a deterministic function of its
//!   arguments. Both are sound because the constrained call pins the result to that deterministic
//!   function of its arguments, so equal (congruent) arguments force an equal result; a
//!   non-deterministic return (e.g. one carrying a fresh witness) is expressible by neither and
//!   stays opaque, and an unconstrained (advice) result is never numbered.
//! - **Conditional Facts:** Assert-derived constants, equalities, and disequalities hold only on
//!   runs that preserve their establishing constraint, so they are exposed only through dedicated
//!   queries (the _Soundness on Rejecting Runs_ section below states the observation model that
//!   makes this style of argument valid). An assert contributes its fact in two directions.
//!
//!   - To every block it strictly _dominates_ it has **already run**, so the constraint is
//!     established there (the `asserted_*` channel; `def(v) dom B dom C` makes scope free).
//!   - To every in-scope block it strictly _post-dominates_ it is **bound to run**: a run visiting
//!     such a block reaches the check — which tests the _genuine_ values — aborts earlier, or
//!     hangs, and every non-reaching outcome rejects (a non-terminating run produces no witness).
//!     This is the `anticipated_*` channel, additionally gated per asserted value on _stability_
//!     **or** finality at the target, and on an explicit in-scope check. Post-dominance, cyclicity,
//!     and finality are measured on the **executable subgraph** when this build's solver pruned a
//!     static edge (honest runs traverse executable edges only; post-dominance as the union with
//!     the static relation — see [`exec_view`]), and on the static CFG otherwise. The in-scope
//!     check stays static dominance always: it guards the _textual_ SSA validity of consumer
//!     rewrites, not a run-time property. See [`stability`].
//!
//!   Within the asserting block both directions apply at index granularity — uses after the assert
//!   via "already ran", uses before it via "bound to run" (same-iteration straight-line, so no
//!   stability gate), while the assert's own operands are informed by neither. Anticipated facts
//!   carry one extra consumer obligation (**Gate 3**): they must never rewrite the operands of any
//!   `Assert`/`AssertCmp`, else two bound-to-run checks of one fact could erase each other and
//!   silently drop the constraint. The anticipated _leader_ (copy-propagation) table exploits this:
//!   since both directions cover every non-assert index, its union equality classes are
//!   index-independent — only each leader carries an in-block availability threshold — and the
//!   own-operands exclusion rests wholly on Gate 3 there. Control-flow derived equality and
//!   disequality facts, being path-conditional, stay dominance-only.
//! - **Witness Forwarding:** Both readings of the one witness↔value correspondence — the
//!   `WriteWitness` hint (`r = witness_of(v)`) and the `ValueOf` projection (`v = value_of(r)`) —
//!   only _add_ constraints and hence never reject an honest run, while two free witnesses are
//!   never unified (the relation is keyed by the witness, and its members are honest values).
//!
//! This analysis **terminates** because every factor has finite height and every loop is monotone,
//! as follows:
//!
//! - **Constants:** The lattice has height 2 (`Top ⊐ Const ⊐ Bottom`) and `set_lattice` only
//!   lowers, so each value changes at most twice.
//! - **Reachability:** `exec_edges` and `reachable` grow monotonically within the finite CFG;
//!   per-block branch facts only ever shrink at joins. The two-worklist loop drains a finite number
//!   of edge/value events.
//! - **Congruence:** Partition refinement only ever _splits_ classes, and the class count is
//!   bounded by the value count, so it stabilises in at most `|values|` rounds.
//! - **Determinism / Interprocedural Summaries:** The determinism bits and the `ReturnJump` lattice
//!   are finite, and a caller is re-queued only when a callee's summary strictly changed, so each
//!   call-graph worklist reaches a fixpoint.
//! - **1-CFA Specialization:** Contexts are `K`-limited call strings over finitely many call sites,
//!   hence finite. Per-context parameter seeds move monotonically down the finite-height constant
//!   lattice, and a context is re-queued only when newly discovered or its seed lowered.
//!
//! # Soundness on Rejecting Runs (the Accept/Reject Model)
//!
//! The conditional-fact arguments above (seen throughout this analysis) lean on phrases like "every
//! non-reaching outcome rejects" and "a block that no accepting run reaches". What follows is a
//! precise statement of what these mean, because a _non-accepting_ run **can** reach such blocks
//! and execute the rewritten code there.
//!
//! The only observable outcome of a run is _accept_ (producing a witness) or _reject_. A failed
//! assert aborts the run and a non-terminating run never finishes, so neither yields a witness, and
//! nothing either computes along the way is observable. A transformation is therefore sound iff it:
//!
//! 1. leaves every accepting run accepting, computing the same results, and
//! 2. leaves every rejecting run rejecting.
//!
//! Obligation 2 explicitly permits changing _how_ a run rejects — aborting at a different
//! constraint, hanging where it used to abort, or vice versa — because all rejections collapse to
//! the same observable. It is discharged by a single invariant: **the establishing assert survives
//! every constraint-preserving rewrite, still checking its genuine (never-rewritten) operand
//! values**. Granted that, consider any run on which an exploited fact is false:
//!
//! - **Dominance Channel:** The rewrite sits strictly after the establisher, so the run aborts _at
//!   the assert_, before the rewritten code — on that run the rewrite is not merely unobservable,
//!   it never executes.
//! - **Anticipated Channel:** The rewrite sits before a bound-to-run establisher, so the run may
//!   compute garbage at the rewrite point — but it then reaches the genuine check and fails it,
//!   aborts on an earlier constraint, or hangs. Every arm rejects, so the garbage never enters an
//!   accepted witness. The same surviving check is what defeats an adversarial witness: an
//!   assignment on which the fact is false still violates that constraint in the transformed
//!   program.
//!
//! The only way to break the invariant is to weaken the establisher _using its own fact_, and the
//! structural gates exist precisely to forbid that. Gate 3 bars anticipated facts from `Assert` and
//! `AssertCmp` operands, and the strict same-block index rules mean an assert never informs its own
//! operands in either direction (see [`conditional`]).
//!
//! Branch folding is the strongest exploitation. Folding a `JmpIf` on an anticipated pin reroutes
//! a doomed run onto a path it never originally took. This stays inside the model because the
//! establisher post-dominates the fold point and hence lies on the _surviving_ path. Thus, the
//! rerouted run still meets the genuine check (possibly later, possibly hanging in a loop instead
//! of aborting — both reject). The fold _can_ even orphan the establisher, but only in one
//! degenerate shape: a check reachable solely through the pruned edge post-dominates the fold
//! point only when every path from the fold point to the exit traverses that edge (the surviving
//! edge can reach it only by looping back through the fold point), so pruning it leaves the exit
//! unreachable from the fold point — every run arriving there hangs, and hanging rejects.
//! Consistently, no accepting run visited the fold point in the original program either: the gated
//! fact holds at every visit, so such a run could never have taken the pruned edge it needed to
//! exit. And when the fact is _true_ on a run, the folded branch is the branch that run took
//! anyway, so a fold never changes where a satisfying run goes — in particular it cannot turn a
//! hang into an accept.
//!
//! "A block that no accepting run reaches" (contradictory asserts under first-writer-wins; see
//! [`conditional`]) is the degenerate case. Contradictory _co-dominating_ asserts mean **no run at
//! all** reaches the block — reaching it requires passing both genuine checks — and contradictory
//! _co-post-dominating_ asserts mean every run visiting it is bound to fail one or the other.
//! Either way, an arbitrary choice among the contradictory facts decorates code whose values appear
//! in no accepted witness.
//!
//! This model settles only the rejecting direction. Obligation 1 — never turning an accepting run
//! rejecting, or changing what it computes — is _not_ "it would fail anyway". Instead it is each
//! channel's own burden to prove its fact _true_ at the rewrite point on every accepting run (the
//! SSA dominance structure for the `asserted_*` channel; stability and scope, Gates 1–2, for the
//! `anticipated_*` channel — see [`conditional`]), which makes the rewrite an identity there.
//!
//! # Deferred Improvements
//!
//! The following are improvements for the future of this analysis. They are either necessary for
//! future work, or deferred because they are expected to yield little benefit on our current
//! circuit corpus.
//!
//! - **Deeper Symbolic Interprocedural Congruence:** The implemented symbolic cross-call congruence
//!   (a callee's return grafted into the caller as a bounded-depth expression over its formals) is
//!   intentionally restricted to a _single_ interprocedural level and a shallow depth cap. A nested
//!   call in a callee's return is an opaque leaf, so a composed gadget `g(x) = h(x) + 1` is not
//!   expressed. Transitively inlining a callee's own symbolic return, or raising the depth cap,
//!   would widen coverage — but multi-level inlining reintroduces a feedback edge into the summary
//!   fixpoint (the projection is currently a structurally output-only post-pass), so it must
//!   restore termination with a depth bound. A further extension would consume the symbolic return
//!   in the constant channel (`eval_call`) to fold a call whose arguments are constant (that
//!   channel is untouched today).

mod conditional;
mod congruence;
mod def_order;
mod exec_view;
mod lattice;
mod solver;
mod stability;
mod summary;

#[cfg(test)]
pub(crate) mod test;

use std::sync::Arc;

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            click_cooper::{
                conditional::ConditionalFacts,
                def_order::DefOrder,
                lattice::Constness,
                solver::{FunctionFacts, solve_with_writeback},
                stability::{BlockReach, CyclicBlocks, static_edge_count},
                summary::{
                    compute_determinism, compute_summaries, compute_sym_summaries, specialize,
                },
            },
            flow_analysis::FlowAnalysis,
            shared::call_string::Context,
            types::TypeInfo,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            BlockId, FunctionId, ProgramPoint, ValueId,
            hlssa::{Constant, HLFunction, HLSSA, HLSSAConstantsSnapshot},
        },
    },
};

// RE-EXPORTS
// ================================================================================================

/// The canonical 1-bit constant for a folded boolean, re-exported for the consumers of this
/// analysis (e.g. the sparse conditional simplification pass).
pub(crate) use self::lattice::bool_constant;

// CLICK COOPER ANALYSIS
// ================================================================================================

/// The Click-Cooper analysis pass.
///
/// See the module documentation for more details.
#[derive(Debug)]
pub struct ClickCooper {
    /// Program-wide interned-constant snapshot.
    consts: HLSSAConstantsSnapshot,

    /// Per-function **intraprocedural** converged facts (parameters and call results `Bottom`).
    functions: HashMap<FunctionId, FunctionFacts>,

    /// Per-`(function, context)` interprocedurally-refined facts (1-CFA), built from the
    /// polymorphic jump-function summaries.
    contexts: HashMap<(FunctionId, Context), FunctionFacts>,

    /// Per-function **conditional** facts.
    ///
    /// They are intentionally disjoint from the unconditional views above.
    conditional: HashMap<FunctionId, ConditionalFacts>,

    /// Per-`(function, context)` **conditional** facts (the 1-CFA refinement of `conditional`).
    ///
    /// Disjoint from `contexts`, exactly as `conditional` is from `functions`.
    conditional_contexts: HashMap<(FunctionId, Context), ConditionalFacts>,
}

impl Analysis for ClickCooper {
    fn dependencies() -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }

    fn compute(ssa: &HLSSA, store: &AnalysisStore) -> Self {
        ClickCooper::run(ssa, store.get::<FlowAnalysis>(), store.get::<TypeInfo>())
    }
}

impl ClickCooper {
    fn run(ssa: &HLSSA, flow: &FlowAnalysis, types: &TypeInfo) -> Self {
        // One snapshot serves all functions: a constant referenced from a function is always
        // interned program-wide.
        let consts = ssa.const_snapshot();

        // Phase 0: per-`(callee, return)` determinism, used below to value-number deterministic
        // static-call results cross-call. It refines only congruence, not reachability and not the
        // constant lattice _directly_ — though a refined congruence can still reach the lattice
        // indirectly through the writeback's `derive_promotions` (a `Cmp{Eq}`/`Cmp{Lt}` over
        // newly-congruent operands folding to a constant). In practice this is corpus-neutral for
        // SCS, but that is an empirically verified result, not a structural guarantee.
        let det = compute_determinism(ssa, flow);

        // Interprocedural summaries first: the polymorphic jump functions, then the symbolic
        // congruence projection as a post-fixpoint pass over the converged summaries). Both are
        // computed before the intraprocedural solve so it can graft symbolic call-return
        // expressions into its congruence partition.
        let (summaries, sym_cache) = compute_summaries(ssa, flow, &consts, &det);
        let sym = compute_sym_summaries(ssa, &consts, &sym_cache);
        drop(sym_cache);

        // One dominance-consistent definition order per function, shared by the congruence-leader
        // finalization (intraprocedural here, per-context inside `specialize`) and every
        // conditional build below: it depends only on function structure and CFG dominance, both
        // context-independent.
        let orders: HashMap<FunctionId, DefOrder> = ssa
            .get_function_ids()
            .map(|fid| {
                (
                    fid,
                    DefOrder::new(ssa.get_function(fid), flow.get_function_cfg(fid)),
                )
            })
            .collect();

        // Static-CFG block cyclicity per function, computed once here (linear per function) for
        // the anticipated stability gate. ClickCooper is its only consumer for this information
        // right now so we avoid doing the extra work in the pipeline-wide CFG builds. This is the
        // _unpruned fast path_: pruned builds derive their own executable view (see `exec_view`).
        let cyclics: HashMap<FunctionId, CyclicBlocks> = ssa
            .get_function_ids()
            .map(|fid| (fid, CyclicBlocks::new(ssa.get_function(fid))))
            .collect();

        // One static-CFG block-reachability table per _cyclic, assert-bearing_ function, shared
        // by every conditional build below whose solver pruned nothing (Part B of the anticipated
        // stability gate is context-independent on that fast path; pruned builds derive their own
        // executable view). Other functions get no entry: an acyclic one has every value stable
        // by rule 2 of the invariance closure, and an assert-free one produces no anticipated
        // fact for the finality gate to test.
        let reaches: HashMap<FunctionId, BlockReach> = ssa
            .get_function_ids()
            .filter(|fid| cyclics[fid].has_cycles() && has_assert_facts(ssa.get_function(*fid)))
            .map(|fid| (fid, BlockReach::new(ssa.get_function(fid))))
            .collect();

        // The deduped static terminator-edge count per function: the baseline of `ExecView`'s
        // unpruned fast path, which compares the solver's executable-edge count against it.
        // Context-independent, so computing it once here spares every per-`(function, context)`
        // conditional build a rebuild of the static edge set.
        let static_edge_counts: HashMap<FunctionId, usize> = ssa
            .get_function_ids()
            .map(|fid| (fid, static_edge_count(ssa.get_function(fid))))
            .collect();

        // One conditional build, shared verbatim by the intraprocedural and per-context loops
        // below; only the converged facts differ between the two.
        let build_conditional = |fid: FunctionId, facts: &FunctionFacts| {
            conditional::build(
                ssa.get_function(fid),
                facts,
                flow.get_function_cfg(fid),
                &consts,
                types.get_function(fid),
                &orders[&fid],
                &cyclics[&fid],
                reaches.get(&fid),
                &det,
                static_edge_counts[&fid],
            )
        };

        // Per-function intraprocedural facts: parameters and call results `Bottom`. The conditional
        // facts are computed from the same converged state plus CFG dominance, kept in a disjoint
        // map so the unconditional view is untouched.
        //
        // The eval-call summary channel MUST stay off here as downstream passes rely on `Call`
        // staying `Bottom`. The _symbolic_ channel is enabled: like the `det` channel above, it
        // refines congruence — not reachability, and not the constant lattice _directly_.
        let mut functions = HashMap::default();
        let mut conditional = HashMap::default();
        for fid in ssa.get_function_ids() {
            let function = ssa.get_function(fid);

            // Combined-fixpoint writeback (intraprocedural): solve constants + reachability, then
            // fold a comparison of congruent operands to a constant and re-solve until no new fact
            // appears (see `solve_with_writeback`). The eval-call `summaries = None` keeps every
            // `Call` result `Bottom` — the contract SCS reads `functions` under — while
            // `Some(&sym)` grafts symbolic call-return expressions into the congruence partition.
            let mut facts = solve_with_writeback(
                function,
                &consts,
                &det,
                None,
                Some(&sym),
                &HashMap::default(),
            );

            // Finalize dominance-aware congruence leaders, so `leader` returns a legal redirect
            // target.
            facts.congruence.compute_leaders(&orders[&fid]);
            conditional.insert(fid, build_conditional(fid, &facts));
            functions.insert(fid, facts);
        }

        // The 1-CFA contextual facts, built from the polymorphic summaries (computed above) plus
        // the symbolic congruence projection.
        let contexts = specialize(ssa, &consts, &summaries, &sym, &det, &orders);

        // The per-`(function, context)` conditional facts, rebuilt by the same `conditional::build`
        // from each context's specialized facts, so every fact is anchored to context-reachable
        // blocks and context-live values — which is what makes the `_in` queries safe to drive
        // context-specialized rewrites, and why they read this map exclusively. Neither view
        // refines the other in general. The map iteration order is immaterial: each build is
        // independent and internally deterministic, and the results land in a keyed map.
        let mut conditional_contexts = HashMap::default();
        for ((fid, ctx), facts) in &contexts {
            conditional_contexts.insert((*fid, ctx.clone()), build_conditional(*fid, facts));
        }

        ClickCooper {
            consts,
            functions,
            contexts,
            conditional,
            conditional_contexts,
        }
    }
}

/// Private query helpers shared by the public query impls.
impl ClickCooper {
    /// The intraprocedural facts of `f`, or `None` if `f` was not analyzed.
    fn facts(&self, f: FunctionId) -> Option<&FunctionFacts> {
        self.functions.get(&f)
    }

    /// The context-specialized (1-CFA) facts of `f` under `ctx`, or `None`.
    fn facts_in(&self, f: FunctionId, ctx: &Context) -> Option<&FunctionFacts> {
        self.contexts.get(&(f, ctx.clone()))
    }

    /// The intraprocedural conditional facts of `f`, or `None` if `f` was not analyzed.
    fn conditional(&self, f: FunctionId) -> Option<&ConditionalFacts> {
        self.conditional.get(&f)
    }

    /// The per-context conditional facts of `f` under `ctx`, or `None`.
    fn conditional_in(&self, f: FunctionId, ctx: &Context) -> Option<&ConditionalFacts> {
        self.conditional_contexts.get(&(f, ctx.clone()))
    }

    /// The constant `v` holds in `facts`, or `None`: interned constants first, then the constant
    /// lattice.
    fn const_in_facts(
        consts: &HLSSAConstantsSnapshot,
        facts: Option<&FunctionFacts>,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        if let Some(c) = consts.get(&v) {
            return Some(c.clone());
        }
        match facts?.lattice.get(&v) {
            // Aggregate (`Blob`) constants are folded only to drive `ArrayGet`/`SliceLen`
            // projections _within_ the analysis; they are never surfaced, so no consumer is asked
            // to materialise an aggregate constant value (only the scalar projections flow out).
            Some(Constness::Const(c)) if c.is_scalar() => Some(c.clone()),
            Some(Constness::Const(_) | Constness::Top | Constness::Bottom) | None => None,
        }
    }
}

/// Unconditional intraprocedural queries — facts that hold in **every** run.
///
/// Using them to replace uses, delete pure defs, or prune unreachable blocks is always correctness
/// preserving.
impl ClickCooper {
    /// The constant `v` provably holds in `f` in _every_ run, or `None`.
    ///
    /// Interned constants and the global constant lattice only; path-sensitive branch facts are
    /// excluded.
    pub fn const_of(&self, f: FunctionId, v: ValueId) -> Option<Arc<Constant>> {
        Self::const_in_facts(&self.consts, self.facts(f), v)
    }

    /// `true` if `bid` reachable in `f`.
    pub fn is_reachable(&self, f: FunctionId, bid: BlockId) -> bool {
        self.facts(f)
            .is_some_and(|facts| facts.reachable.contains(&bid))
    }

    /// `true` if the CFG edge `from -> to` proven executable in `f`.
    pub fn is_executable_edge(&self, f: FunctionId, from: BlockId, to: BlockId) -> bool {
        self.facts(f)
            .is_some_and(|facts| facts.exec_edges.contains(&(from, to)))
    }

    /// Every value proven (unconditionally) constant in `f`, as `(value, constant)` pairs.
    ///
    /// Excludes already-interned constant values and conditional branch facts.
    pub fn new_const_values(&self, f: FunctionId) -> Vec<(ValueId, Arc<Constant>)> {
        let Some(facts) = self.facts(f) else {
            return Vec::new();
        };
        facts
            .lattice
            .iter()
            .filter_map(|(v, e)| match e {
                // Surface scalars only; aggregate (`Blob`) constants stay internal to the analysis
                // (see `const_in_facts`).
                Constness::Const(c) if c.is_scalar() => Some((*v, c.clone())),
                Constness::Const(_) | Constness::Top | Constness::Bottom => None,
            })
            .collect()
    }

    /// `true` if `a` and `b` are proven _structurally_ congruent in `f`.
    pub fn known_equal(&self, f: FunctionId, a: ValueId, b: ValueId) -> bool {
        self.facts(f)
            .is_some_and(|facts| facts.congruence.known_equal(a, b))
    }

    /// Every value structurally congruent to `v` in `f` (including `v`), sorted by value id.
    pub fn congruence_class(&self, f: FunctionId, v: ValueId) -> Vec<ValueId> {
        self.facts(f)
            .map(|facts| facts.congruence.class_members(v).to_vec())
            .unwrap_or_default()
    }

    /// The leader of `v`'s congruence class in `f`: the root-most congruent member whose definition
    /// dominates `v`'s definition (`v` itself when no other member does).
    pub fn leader(&self, f: FunctionId, v: ValueId) -> Option<ValueId> {
        self.facts(f).and_then(|facts| facts.congruence.leader(v))
    }
}

/// Unconditional interprocedural queries — the 1-CFA per-`(function, context)` refinement.
///
/// These read the specialized `contexts` map (never the `functions` view), so they are disjoint
/// from the intraprocedural queries above. They are unconditional facts _within_ their context: a
/// caller's argument constants seed the callee's parameters, constant-returning calls fold, and the
/// combined-fixpoint writeback folds comparisons of congruent operands per context.
///
/// A consumer must respect two contracts these facts inherit. First, a `const_of_in` / `leader_in`
/// answer is conditional on the context `ctx`, so it must drive a context-specialized rewrite only
/// — never lift it into a context-independent edit. Second, when a writeback-folded constant is a
/// `WitnessOf`-typed comparison result, the substitution must keep it witness-typed.
impl ClickCooper {
    /// The constant `v` provably holds in `f` under calling context `ctx`, or `None`.
    pub fn const_of_in(&self, f: FunctionId, ctx: &Context, v: ValueId) -> Option<Arc<Constant>> {
        Self::const_in_facts(&self.consts, self.facts_in(f, ctx), v)
    }

    /// `true` if `a` and `b` are proven structurally congruent in `f` under context `ctx`.
    pub fn known_equal_in(&self, f: FunctionId, ctx: &Context, a: ValueId, b: ValueId) -> bool {
        self.facts_in(f, ctx)
            .is_some_and(|facts| facts.congruence.known_equal(a, b))
    }

    /// Every value congruent to `v` in `f` under context `ctx` (including `v`), sorted by value id.
    pub fn congruence_class_in(&self, f: FunctionId, ctx: &Context, v: ValueId) -> Vec<ValueId> {
        self.facts_in(f, ctx)
            .map(|facts| facts.congruence.class_members(v).to_vec())
            .unwrap_or_default()
    }

    /// The leader of `v`'s congruence class in `f` under context `ctx` (a legal redirect target, as
    /// in [`Self::leader`]).
    pub fn leader_in(&self, f: FunctionId, ctx: &Context, v: ValueId) -> Option<ValueId> {
        self.facts_in(f, ctx)
            .and_then(|facts| facts.congruence.leader(v))
    }

    /// The contexts `f` was specialized in (sorted), for enumerating its per-context facts.
    pub fn contexts_of(&self, f: FunctionId) -> Vec<Context> {
        let mut out: Vec<Context> = self
            .contexts
            .keys()
            .filter(|(g, _)| *g == f)
            .map(|(_, ctx)| ctx.clone())
            .collect();
        out.sort();
        out
    }
}

/// Intraprocedural conditional queries — facts established by a branch or assert.
///
/// They are sound _only_ under constraint-preserving use: a local, structure-preserving rewrite
/// that keeps the establishing branch/assert (never folding away the constraint that proves the
/// fact).
///
/// These are **intraprocedural**: a conditional fact here is established by control flow _internal
/// to `f`_ (a dominating assert, a branch predicate, a witness write), so the fact itself holds in
/// _every_ calling context. Reading them is sound for **context-independent** rewrites.
///
/// A **context-specialized** rewrite — one applied to a per-context clone that prunes context-dead
/// blocks — must use the `_in` analogs instead (rebuilt per `(function, context)` from the
/// specialized facts; see [`Self::asserted_const_in`] and its sibling `_in` queries). This is
/// because an intraprocedural fact can name a value whose definition the clone prunes (e.g. a
/// `ValueOf`-derived witness-forward member defined in a context-dead block).
///
/// **No pointwise inclusion holds between the two views, in either direction, on any channel.**
/// Context seeding usually adds facts (a pinned parameter turns an asserted equality into an
/// asserted constant; a pruned edge creates a disequality) but can also remove them. The following
/// are examples:
///
/// - An equality migrates to the const channel when one side becomes a per-context constant.
/// - An assert whose _asserted_ side becomes a per-context constant moves to the (strictly
///   stronger) unconditional channel and out of `asserted_const_in`.
/// - An aggregate-pinning assert contributes nothing.
/// - A per-context constant can _split_ an intraprocedurally-proven congruence so a branch the
///   intraprocedural writeback folded stays live in the context — shifting reachability, and every
///   reachability-derived fact, in the _growing_ direction.
///
/// A consumer wanting the full per-context picture must consult both families, applying each fact
/// only under its own contract.
impl ClickCooper {
    /// The constant `v` holds on entry to `bid` in `f`, honoring path-sensitive branch facts.
    pub fn const_in_block(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<Arc<Constant>> {
        if let Some(c) = self.consts.get(&v) {
            return Some(c.clone());
        }
        let facts = self.facts(f)?;
        if let Some(known) = facts.block_facts.get(&bid).and_then(|m| m.get(&v)) {
            return Some(bool_constant(*known));
        }
        Self::const_in_facts(&self.consts, Some(facts), v)
    }

    /// `Some(...)` if `v` known true/false on entry to `bid` in `f` (a branch predicate fact).
    pub fn bool_fact(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<bool> {
        self.facts(f)?.block_facts.get(&bid)?.get(&v).copied()
    }

    /// If `v` is an (in-block) constant boolean, get its value, honoring branch facts, or `None`
    /// otherwise.
    pub fn const_bool_in_block(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<bool> {
        self.const_in_block(f, bid, v)
            .and_then(|c| lattice::const_bool(&c))
    }

    /// The branch predicate facts at entry to `bid` in `f`, as `(value, bool-constant)` pairs
    /// sorted by value id.
    pub fn block_bool_facts(&self, f: FunctionId, bid: BlockId) -> Vec<(ValueId, Arc<Constant>)> {
        let Some(facts) = self.facts(f) else {
            return Vec::new();
        };
        let Some(m) = facts.block_facts.get(&bid) else {
            return Vec::new();
        };
        let mut out: Vec<(ValueId, Arc<Constant>)> =
            m.iter().map(|(v, b)| (*v, bool_constant(*b))).collect();
        out.sort_by_key(|(v, _)| v.0);
        out
    }

    /// The constant `v` is pinned to at `point` in `f` (`Assert{v}`⇒`true`, or
    /// `AssertCmp{Eq, v, c}`) by a _dominating assert_ (which holds at every index of `point.block`)
    /// or by an assert *earlier in `point.block`* than `point.index`.
    ///
    /// `point.index` is the position of the _using_ instruction itself within
    /// `point.block.get_instructions()` — a consumer walking `get_instructions().enumerate()` has
    /// it for free, and a use in the terminator takes `index == get_instructions().count()`. A
    /// same-block assert is reported only for use-indices _strictly greater_ than its own, so the
    /// assert never folds its own operand and vacuums the constraint.
    ///
    /// A conditional fact, deliberately disjoint from [`Self::const_of`] as a consumer must keep
    /// the establishing assert. The constant is always scalar: an assert pinning a value to an
    /// aggregate contributes no fact, upholding the module contract that `Blob` constants are
    /// never surfaced.
    pub fn asserted_const(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        self.conditional(f)?.asserted_const(point, v)
    }

    /// `true` if `a == b` is proven at `point` in `f` by a dominating `AssertCmp{Eq}` or one
    /// earlier in `point.block` than `point.index`.
    ///
    /// The conditional analog of [`Self::known_equal`] (structural congruence): sound only for
    /// constraint-preserving use, so kept out of the unconditional partition.
    pub fn asserted_equal(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        a: ValueId,
        b: ValueId,
    ) -> bool {
        self.conditional(f)
            .is_some_and(|c| c.asserted_equal(point, a, b))
    }

    /// Gets the canonical representative of `v`'s transitive asserted-equal class at `point` in
    /// `f`.
    ///
    /// This is the dominance-root-most member, a value provably equal to `v` there whose definition
    /// dominates `point` — or `None` if `v` is in no asserted equality at `point`. May return `v`
    /// itself when `v` is already the root-most (a consumer copy-propagates only when the leader
    /// differs from `v`).
    ///
    /// The conditional analog of the congruence [`Self::leader`]: sound only under
    /// constraint-preserving use.
    pub fn asserted_leader(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<ValueId> {
        self.conditional(f)?.asserted_leader(point, v)
    }

    /// `true` if `a` and `b` are proven unequal on entry to `bid` in `f` (the false edge of an
    /// equality branch).
    ///
    /// Note that this only accounts for _disequality_ `a != b`, and not expanded linear
    /// inequalities.
    pub fn known_unequal(&self, f: FunctionId, bid: BlockId, a: ValueId, b: ValueId) -> bool {
        self.conditional(f)
            .is_some_and(|c| c.known_disequal(bid, a, b))
    }

    /// The honest values the free witness handle `r` is equal to in `f`, containing no duplicates.
    ///
    ///
    /// It is drawn from both readings of one correspondence: the unpinned
    /// `WriteWitness{result: Some(r), value, pinned: false}` hint (`r = witness_of(value)`) and
    /// every `Cast{value: r, target: ValueOf}` read (`result = value_of(r)`).
    ///
    /// A sound _redirect_ fact (each member adds a functional constraint ⇒ `A_M ⊆ A_N`, never
    /// rejecting the honest run), not a structural congruence — so it lives here, never in
    /// [`Self::known_equal`], and two free witnesses are never unified.
    pub fn witness_forward(&self, f: FunctionId, r: ValueId) -> &[ValueId] {
        self.conditional(f)
            .map(|c| c.witness_forward(r))
            .unwrap_or_default()
    }

    /// The constant `v` is pinned to at `point` in `f` by a _bound-to-run_ assert: one in a block
    /// strictly post-dominating `point.block` or one _later_ in `point.block` than `point.index`
    /// (the index convention of [`Self::asserted_const`], mirrored).
    ///
    /// The anticipated (post-dominance) counterpart of [`Self::asserted_const`], and disjoint from
    /// it. This never reports a dominance fact, so a consumer wanting both directions chains the
    /// two queries.
    ///
    /// On top of constraint-preserving use, an anticipated fact carries one extra obligation: it is
    /// justified by the assert _still running_ on the genuine values — a run on which the fact is
    /// false aborts there (or hangs, producing no witness) — so a consumer must **never** use it to
    /// rewrite the operands of any `Assert`/`AssertCmp`. Two bound-to-run checks of the same fact
    /// could otherwise erase each other, silently dropping the constraint.
    pub fn anticipated_const(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        self.conditional(f)?.anticipated_const(point, v)
    }

    /// [`Self::anticipated_const`] narrowed to a boolean pin: `Some(b)` when a bound-to-run assert
    /// pins `v` to the boolean constant `b` at `point` (e.g. an `Assert{v}` pinning `v = true`),
    /// for consumers deciding a branch. Same contract as [`Self::anticipated_const`].
    pub fn anticipated_const_bool(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<bool> {
        self.anticipated_const(f, point, v)
            .and_then(|c| lattice::const_bool(&c))
    }

    /// `true` if `a == b` is proven at `point` in `f` by a bound-to-run `AssertCmp{Eq}`: one in a
    /// strictly post-dominating block (values single-valued per invocation or final at the block,
    /// and in scope), or one later in `point.block`.
    ///
    /// The anticipated counterpart of [`Self::asserted_equal`], disjoint from it and under the
    /// [`Self::anticipated_const`] contract (never rewrite `Assert`/`AssertCmp` operands with it).
    pub fn anticipated_equal(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        a: ValueId,
        b: ValueId,
    ) -> bool {
        self.conditional(f)
            .is_some_and(|c| c.anticipated_equal(point, a, b))
    }

    /// The canonical representative of `v`'s transitive equality class at `point` in `f` over the
    /// union of the block's dominance-direction, same-block (both index directions, so a use
    /// _before_ a same-block assert redirects too), and anticipated pairs — or `None` if `v` joins
    /// no class.
    ///
    /// Unlike the boolean queries this is a _union_ view (a single redirect target must account
    /// for every equality holding at `point`), so it supersedes [`Self::asserted_leader`] wherever
    /// the anticipated direction contributes and falls back to it elsewhere. The leader's
    /// definition dominates `point` — enforced within the block by a per-leader index threshold
    /// that any query at a real use of `v` passes — making it a legal redirect target. A redirect
    /// it produces may be anticipated-justified, so it inherits the [`Self::anticipated_const`]
    /// contract (never into `Assert`/`AssertCmp` operands) — the sole guard at the establishing
    /// assert's own index.
    pub fn anticipated_leader(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<ValueId> {
        self.conditional(f)?.anticipated_leader(point, v)
    }
}

/// Interprocedural conditional queries, the _branch-fact_ family — the per-context analogs of the
/// branch-fact intraprocedural conditional queries above (e.g. for a clone of `f` specialized to
/// `ctx`).
///
/// They are sound _only_ under constraint-preserving use: a local, structure-preserving rewrite
/// that keeps the establishing branch (never folding away the constraint that proves the fact).
impl ClickCooper {
    /// [`Self::const_in_block`] under calling context `ctx`.
    ///
    /// Honors the context-specialized branch facts, so a context that folds more branches is more
    /// precise.
    pub fn const_in_block_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        bid: BlockId,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        if let Some(c) = self.consts.get(&v) {
            return Some(c.clone());
        }
        let facts = self.facts_in(f, ctx)?;
        if let Some(known) = facts.block_facts.get(&bid).and_then(|m| m.get(&v)) {
            return Some(bool_constant(*known));
        }
        Self::const_in_facts(&self.consts, Some(facts), v)
    }

    /// [`Self::bool_fact`] under calling context `ctx`.
    pub fn bool_fact_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        bid: BlockId,
        v: ValueId,
    ) -> Option<bool> {
        self.facts_in(f, ctx)?
            .block_facts
            .get(&bid)?
            .get(&v)
            .copied()
    }

    /// [`Self::const_bool_in_block`] under calling context `ctx`.
    pub fn const_bool_in_block_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        bid: BlockId,
        v: ValueId,
    ) -> Option<bool> {
        self.const_in_block_in(f, ctx, bid, v)
            .and_then(|c| lattice::const_bool(&c))
    }

    /// [`Self::block_bool_facts`] under calling context `ctx`.
    pub fn block_bool_facts_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        bid: BlockId,
    ) -> Vec<(ValueId, Arc<Constant>)> {
        let Some(facts) = self.facts_in(f, ctx) else {
            return Vec::new();
        };
        let Some(m) = facts.block_facts.get(&bid) else {
            return Vec::new();
        };
        let mut out: Vec<(ValueId, Arc<Constant>)> =
            m.iter().map(|(v, b)| (*v, bool_constant(*b))).collect();
        out.sort_by_key(|(v, _)| v.0);
        out
    }
}

/// Interprocedural conditional queries, the _assert/disequality/witness_ family — reading the
/// `ConditionalFacts` rebuilt per `(function, context)` from the context-specialized facts.
///
/// They inherit **both** contracts.
///
/// - **Constraint-Preserving Use:** A local, structure-preserving rewrite that keeps the
///   establishing assert/branch (never folding away the constraint that proves the fact).
/// - **Context-Locality:** An answer that is conditional on `ctx` must drive a context-specialized
///   rewrite only — never be lifted into a context-independent change.
///
/// The context refinement _typically_ strengthens the assert-const and disequality channels (e.g.
/// an `AssertCmp{Eq, x, p}` whose `p` is a per-context constant pins `x` to an asserted _constant_
/// where the intraprocedural view only has an asserted _equality_), but no channel is a pointwise
/// superset or subset of its intraprocedural counterpart. These queries read the per-context map
/// exclusively: its facts are anchored to context-live blocks and values, which is what makes them
/// safe to drive context-specialized rewrites.
impl ClickCooper {
    /// [`Self::asserted_const`] under calling context `ctx`.
    pub fn asserted_const_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        self.conditional_in(f, ctx)?.asserted_const(point, v)
    }

    /// [`Self::asserted_equal`] under calling context `ctx`.
    pub fn asserted_equal_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        point: ProgramPoint,
        a: ValueId,
        b: ValueId,
    ) -> bool {
        self.conditional_in(f, ctx)
            .is_some_and(|c| c.asserted_equal(point, a, b))
    }

    /// [`Self::asserted_leader`] under calling context `ctx`.
    pub fn asserted_leader_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<ValueId> {
        self.conditional_in(f, ctx)?.asserted_leader(point, v)
    }

    /// [`Self::known_unequal`] under calling context `ctx`.
    pub fn known_unequal_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        bid: BlockId,
        a: ValueId,
        b: ValueId,
    ) -> bool {
        self.conditional_in(f, ctx)
            .is_some_and(|c| c.known_disequal(bid, a, b))
    }

    /// [`Self::witness_forward`] under calling context `ctx`.
    ///
    /// Reads the per-context scan exclusively: a forward established in a context-dead block is
    /// dropped (so every member is context-live), while a block live only under the context's
    /// refinement can contribute a forward the intraprocedural set lacks — neither set contains
    /// the other in general.
    pub fn witness_forward_in(&self, f: FunctionId, ctx: &Context, r: ValueId) -> &[ValueId] {
        self.conditional_in(f, ctx)
            .map(|c| c.witness_forward(r))
            .unwrap_or_default()
    }

    /// [`Self::anticipated_const`] under calling context `ctx` (inheriting its never-into-assert-
    /// operands obligation on top of the two contracts above).
    pub fn anticipated_const_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        self.conditional_in(f, ctx)?.anticipated_const(point, v)
    }

    /// [`Self::anticipated_equal`] under calling context `ctx` (same inherited obligations as
    /// [`Self::anticipated_const_in`]).
    pub fn anticipated_equal_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        point: ProgramPoint,
        a: ValueId,
        b: ValueId,
    ) -> bool {
        self.conditional_in(f, ctx)
            .is_some_and(|c| c.anticipated_equal(point, a, b))
    }

    /// [`Self::anticipated_leader`] under calling context `ctx` (same inherited obligations as
    /// [`Self::anticipated_const_in`]).
    pub fn anticipated_leader_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<ValueId> {
        self.conditional_in(f, ctx)?.anticipated_leader(point, v)
    }
}

// INTERNAL FUNCTIONS
// ================================================================================================

/// `true` if `function` contains any fact-establishing assert.
///
/// The gate for building its [`BlockReach`], which only the anticipated channel's finality gate
/// consults.
fn has_assert_facts(function: &HLFunction) -> bool {
    function
        .get_blocks()
        .any(|(_, block)| block.get_instructions().any(|instr| instr.is_assert()))
}
