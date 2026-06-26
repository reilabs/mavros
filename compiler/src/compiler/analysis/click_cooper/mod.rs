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
//!   with the caller's argument constants, plus determinism-gated *cross-call congruence* (two
//!   constrained static calls to one callee with congruent arguments yield congruent results).
//! - **Conditional facts** are ones derived from asserts, equalities, disequalities, and unpinned
//!   witness forwarding. They are computed post-convergence over the same reachability state plus
//!   CFG dominance, and are intentionally disjoint from the unconditional view.
//!
//! **Unconditional facts** are those which hold on any path so that any use (replacing a use,
//! deleting a pure definition, or pruning an unreachable block) maintains soundness. **Conditional
//! facts** are those established through control flow (branches, assertions, witness operations)
//! that are only correct to use in contexts where the establishing constraints are preserved.
//!
//! # Combined-Fixpoint Writeback
//!
//! The constant and reachability factors feed congruence (constants seed the partition,
//! reachability scopes φ-operands), and congruence feeds *back*: a comparison whose operands are
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
//!   the meet of the argument constants over *every* static call path mapped to that context, so a
//!   per-context constant holds on every concrete invocation in that context.
//! - **Cross-Call Congruence:** A constrained static call's result is value-numbered by
//!   `(callee, return-index)` over its argument classes only when the callee's return is a
//!   deterministic function of its arguments; equal arguments then force an equal result. A
//!   non-deterministic return (e.g. one carrying a fresh witness) stays opaque.
//! - **Conditional Facts:** Assert-derived constants, equalities, and disequalities hold only on
//!   accepting runs that preserve their establishing constraint, so they are exposed only through
//!   dedicated queries. An assert, being a global constraint that holds at every point of an
//!   accepting run, is *also* attributed to in-scope blocks it post-dominates (the assert is then
//!   guaranteed on every continuation), and — at index granularity — to uses *after* it in its own
//!   block (the index-precise "already ran" direction; a use textually before a same-block assert
//!   still sees only the cross-block facts). Control-flow derived equality and disequality facts,
//!   being path-conditional, stay dominance-only.
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
//! - **Congruence:** Partition refinement only ever *splits* classes, and the class count is
//!   bounded by the value count, so it stabilises in at most `|values|` rounds.
//! - **Determinism / Interprocedural Summaries:** The determinism bits and the `ReturnJump` lattice
//!   are finite, and a caller is re-queued only when a callee's summary strictly changed, so each
//!   call-graph worklist reaches a fixpoint.
//! - **1-CFA Specialization:** Contexts are `K`-limited call strings over finitely many call sites,
//!   hence finite. Per-context parameter seeds move monotonically down the finite-height constant
//!   lattice, and a context is re-queued only when newly discovered or its seed lowered.
//!
//! # Deferred Improvements
//!
//! The following are improvements planned for the future of this analysis.
//!
//! - **Symbolic Interprocedural Congruence:** Cross-call congruence currently requires that *all*
//!   arguments are congruent and is gated on a whole-return determinism bit; it does not graft a
//!   callee's return expression into the caller, so it cannot relate `f(x)` to an open expression
//!   over `x`, nor see that a return ignores some argument. A symbolic return jump function would
//!   recover both.
//! - **Per-context Conditional Facts:** The assert/disequality/witness-forwarding conditional facts
//!   are computed once per function and not refined per 1-CFA context, so they have no `_in(f, ctx,
//!   …)` query: a caller reads them through the intraprocedural methods, whose (sound,
//!   context-independent) answer is a per-context under-approximation. Recomputing the conditional
//!   pass per `(function, context)` — as `specialize` already does for the unconditional facts —
//!   would let a context-specialized assert or branch expose more facts, and would warrant adding
//!   the `_in` variants.
//! - **Sound Post-Dominance for Assert Facts:** Assert facts (`asserted_const`/`asserted_equal`)
//!   currently fan out by dominance only. The post-dominance direction — an assert in `B` is *bound
//!   to run* at every block `C` it strictly post-dominates, so its fact would also hold at a use
//!   *before* a requires an additional notion of value stability to be added to the analysis. A
//!   future post-dominance direction would require gating all of the asserted values on being
//!   value-stable using a cyclic-SCC membership primitive.

mod conditional;
mod congruence;
mod lattice;
mod solver;
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
                lattice::{Constness, bool_constant},
                solver::{FunctionFacts, solve_with_writeback},
                summary::{compute_determinism, compute_summaries, specialize},
            },
            flow_analysis::FlowAnalysis,
            shared::call_string::Context,
            types::TypeInfo,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            BlockId, FunctionId, ProgramPoint, ValueId,
            hlssa::{Constant, HLSSA, HLSSAConstantsSnapshot},
        },
    },
};

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
        // static-call results cross-call. It only refines congruence (never the constant lattice or
        // reachability), so it leaves the SCCP-visible facts byte-identical.
        let det = compute_determinism(ssa, flow);

        // Per-function intraprocedural facts: parameters and call results `Bottom` (no summaries
        // here). The conditional side facts are computed from the same converged state plus CFG
        // dominance, kept in a disjoint map so the unconditional view is untouched.
        //
        // This solve MUST stay summary-free: SCCP reads `functions` (via `const_of`/
        // `new_const_values`/`is_reachable`) and relies on `Call` staying `Bottom`, so that every
        // constant-valued result it aliases or deletes is a pure scalar fold. Wiring summaries in
        // here would let a constrained `Call` result become `Const` and break that contract (an
        // `assert!` in SCCP's `rewrite` guards against it).
        let mut functions = HashMap::default();
        let mut conditional = HashMap::default();
        for fid in ssa.get_function_ids() {
            let function = ssa.get_function(fid);

            // Combined-fixpoint writeback (intraprocedural, summary-free): solve constants +
            // reachability, then fold a comparison of congruent operands to a constant and re-solve
            // until no new fact appears (see `solve_with_writeback`). `summaries = None` keeps every
            // `Call` result `Bottom` — the contract SCCP reads `functions` under — and round 0 has
            // empty promotions, so a function with no such comparison is solved once and unchanged.
            let mut facts =
                solve_with_writeback(function, &consts, &det, None, &HashMap::default());

            // Finalize dominance-aware congruence leaders against the function's CFG, so `leader`
            // returns a legal redirect target.
            facts
                .congruence
                .compute_leaders(function, flow.get_function_cfg(fid));
            let cond = conditional::build(
                function,
                &facts,
                flow.get_function_cfg(fid),
                &consts,
                types.get_function(fid),
            );
            conditional.insert(fid, cond);
            functions.insert(fid, facts);
        }

        // Handle the interprocedural layer using polymorphic jump-function summaries then
        // contextual facts on the 1-CFA.
        let summaries = compute_summaries(ssa, flow, &consts, &det);
        let contexts = specialize(ssa, flow, &consts, &summaries, &det);

        ClickCooper {
            consts,
            functions,
            contexts,
            conditional,
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
            // projections *within* the analysis; they are never surfaced, so no consumer is asked
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
    /// The constant `v` provably holds in `f` in *every* run, or `None`.
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

    /// `true` if `a` and `b` are proven *structurally* congruent in `f`.
    pub fn known_equal(&self, f: FunctionId, a: ValueId, b: ValueId) -> bool {
        self.facts(f)
            .is_some_and(|facts| facts.congruence.known_equal(a, b))
    }

    /// Every value structurally congruent to `v` in `f` (including `v`), sorted by value id.
    pub fn congruence_class(&self, f: FunctionId, v: ValueId) -> Vec<ValueId> {
        self.facts(f)
            .map(|facts| facts.congruence.class_members(v))
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
/// These read the specialized `contexts` map (never the SCCP-visible `functions` view), so they are
/// disjoint from the intraprocedural queries above. They are unconditional facts *within* their
/// context: a caller's argument constants seed the callee's parameters, constant-returning calls
/// fold, and the combined-fixpoint writeback folds comparisons of congruent operands per context.
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
            .map(|facts| facts.congruence.class_members(v))
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
/// They are sound *only* under constraint-preserving use: a local, structure-preserving rewrite
/// that keeps the establishing branch/assert (never folding away the constraint that proves the
/// fact).
///
/// These are **intraprocedural** and hence have no context. A conditional fact is established by
/// control flow *internal to `f`* (a dominating assert, a branch predicate, a witness write), so it
/// holds in *every* calling context. A context only adds parameter constants and prunes blocks, so
/// it can make a block unreachable but can never invalidate a fact on a block that remains
/// reachable. The intraprocedural answer is thus a sound **under-approximation** of the per-context
/// one.
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
    /// `AssertCmp{Eq, v, c}`) by a *dominating assert* (which holds at every index of `point.block`)
    /// or by an assert *earlier in `point.block`* than `point.index`.
    ///
    /// `point.index` is the position of the *using* instruction itself within
    /// `point.block.get_instructions()` — a consumer walking `get_instructions().enumerate()` has
    /// it for free, and a use in the terminator takes `index == get_instructions().count()`. A
    /// same-block assert is reported only for use-indices *strictly greater* than its own, so the
    /// assert never folds its own operand and vacuums the constraint.
    ///
    /// A conditional fact, deliberately disjoint from [`Self::const_of`] as a consumer must keep
    /// the establishing assert.
    pub fn asserted_const(
        &self,
        f: FunctionId,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        self.conditional.get(&f)?.asserted_const(point, v)
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
        self.conditional
            .get(&f)
            .is_some_and(|c| c.asserted_equal(point, a, b))
    }

    /// `true` if `a` and `b` are proven unequal on entry to `bid` in `f` (the false edge of an
    /// equality branch).
    ///
    /// Note that this only accounts for _disequality_ `a != b`, and not expanded linear
    /// inequalities.
    pub fn known_unequal(&self, f: FunctionId, bid: BlockId, a: ValueId, b: ValueId) -> bool {
        self.conditional
            .get(&f)
            .is_some_and(|c| c.known_disequal(bid, a, b))
    }

    /// The honest values the free witness handle `r` is equal to in `f`, containing no duplicates.
    ///
    ///
    /// It is drawn from both readings of one correspondence: the unpinned
    /// `WriteWitness{result: Some(r), value, pinned: false}` hint (`r = witness_of(value)`) and
    /// every `Cast{value: r, target: ValueOf}` read (`result = value_of(r)`).
    ///
    /// A sound *redirect* fact (each member adds a functional constraint ⇒ `A_M ⊆ A_N`, never
    /// rejecting the honest run), not a structural congruence — so it lives here, never in
    /// [`Self::known_equal`], and two free witnesses are never unified.
    pub fn witness_forward(&self, f: FunctionId, r: ValueId) -> &[ValueId] {
        self.conditional
            .get(&f)
            .map(|c| c.witness_forward(r))
            .unwrap_or_default()
    }
}

/// Interprocedural conditional queries — the per-context analogs of the *branch-fact*
/// intraprocedural conditional queries above (e.g. for a clone of `f` specialized to `ctx`).
///
/// They are sound *only* under constraint-preserving use: a local, structure-preserving rewrite
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
