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
//! # Correctness
//!
//! This analysis is **sound** because each fact class is sound, with the reasoning given as
//! follows:
//!
//! - **Constants:** The transfer functions fold a value only when the result is exact for the
//!   operand widths so `value == c` holds under any advice.
//! - **Reachability:** An edge is executable only when a predecessor's terminator can take it. A
//!   block proven unreachable is thus taken in no run.
//! - **Congruence:** Two values are congruent iff computed by the same operator from operand-wise
//!   congruent inputs, hence equal in all runs. Congruence never asserts an equality an adversarial
//!   witness could validate.
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
//!   guaranteed on every continuation). Control-flow derived equality and disequality facts, being
//!   path-conditional, stay dominance-only.
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
//! - **Combined-Fixpoint Writebacks:** Congruence runs as a single pass over the converged
//!   reachability rather than interleaved with the constant worklist, because the reverse couplings
//!   are not wired: a congruence class with a `Const` member does not promote its peers into the
//!   (SCCP-visible) constant lattice, and a branch condition congruent to a constant does not fold
//!   an edge. These pay off only in a consumer that exploits the full combined fixpoint; until such
//!   a consumer exists, omitting them keeps SCCP's output unchanged. A future consumer can recover
//!   the promotion by composing [`ClickCooper::known_equal`] with [`ClickCooper::const_of`].
//! - **Assert Facts at Index Granularity:** Assert-derived facts are attributed at block-entry
//!   granularity via *strict* dominance/post-dominance, so a use in the asserting block itself
//!   (after the assert) is not yet claimed. Index-precise program points would recover it; soundness
//!   is unaffected.
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
//! - **Aggregate Constant Folding:** The pure, value-semantic sequence ops (`MkSeq`, `MkRepeated`,
//!   `MkSeqOfBlob`, `ArrayGet`, `ArraySet`, `SlicePush`, `SliceLen`) are value-numbered (see
//!   `congruence::op_signature`) but never constant-folded. The `Constness` lattice carries only a
//!   scalar `Arc<Constant>` and the fold functions are scalar, so it cannot represent an aggregate
//!   constant. Future work can widen the lattice to handle aggregate constants and transfer
//!   functions.

mod conditional;
mod congruence;
mod lattice;
mod solver;
mod summary;

use std::sync::Arc;

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            click_cooper::{
                conditional::ConditionalFacts,
                lattice::{Constness, bool_constant},
                solver::{FunctionFacts, FunctionSolver},
                summary::{compute_determinism, compute_summaries, specialize},
            },
            flow_analysis::FlowAnalysis,
            shared::call_string::Context,
            types::TypeInfo,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            BlockId, FunctionId, ValueId,
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
            let mut solver = FunctionSolver::new(function, &consts).with_determinism(&det);
            solver.run();
            let mut facts = solver.into_facts();

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
            Some(Constness::Const(c)) => Some(c.clone()),
            Some(Constness::Top) | Some(Constness::Bottom) | None => None,
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
                Constness::Const(c) => Some((*v, c.clone())),
                Constness::Top | Constness::Bottom => None,
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
/// context: a caller's argument constants seed the callee's parameters, and constant-returning
/// calls fold.
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

    /// The constant `v` is pinned to on entry to `bid` in `f` by a *dominating assert*
    /// (`Assert{v}`⇒`true`, or `AssertCmp{Eq, v, c}`).
    ///
    /// A conditional fact, deliberately disjoint from [`Self::const_of`] as a
    /// consumer must keep the establishing assert.
    pub fn asserted_const(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<Arc<Constant>> {
        self.conditional.get(&f)?.asserted_const(bid, v)
    }

    /// `true` if a dominating `AssertCmp{Eq}` proves `a == b` on entry to `bid` in `f`.
    ///
    /// The conditional analog of [`Self::known_equal`] (structural congruence): sound only for
    /// constraint-preserving use, so kept out of the unconditional partition.
    pub fn asserted_equal(&self, f: FunctionId, bid: BlockId, a: ValueId, b: ValueId) -> bool {
        self.conditional
            .get(&f)
            .is_some_and(|c| c.asserted_equal(bid, a, b))
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

#[cfg(test)]
pub(crate) mod tests {
    use super::{ClickCooper, Context, FlowAnalysis};
    use crate::compiler::{
        Field,
        analysis::types::Types,
        ssa::{
            Terminator,
            hlssa::{
                BinaryArithOpKind, CallTarget, CastTarget, CmpKind, Constant, HLSSA, OpCode,
                ScalarFold, SequenceTargetType, SliceOpDir, Type,
            },
        },
    };

    /// Build the analysis for `ssa` with freshly-computed dependencies, for test use only.
    pub(crate) fn run_in_test(ssa: &HLSSA) -> ClickCooper {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        ClickCooper::run(ssa, &flow, &types)
    }

    /// `OpCode::scalar_fold` is the single source of truth for "foldable *scalar* op", and value
    /// numbering is a strict superset of it.
    ///
    /// This locks the projections that read both: `is_pure_scalar_fold` agrees with
    /// `scalar_fold().is_some()`; value numbering (`pure_op_operands`, backed by `op_signature`)
    /// fires on every foldable scalar op *except* witness casts, **and additionally** on the pure
    /// sequence ops, which are not scalar-foldable. Both deliberate asymmetries are asserted below
    /// so neither can be silently dropped.
    #[test]
    fn scalar_fold_is_the_single_classifier() {
        use super::congruence::pure_op_operands;

        let ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let v = || ssa.fresh_value();

        // One instance of every foldable scalar op, each paired with the `ScalarFold` variant it
        // must decompose to.
        let foldable: Vec<OpCode> = vec![
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: v(),
                lhs: v(),
                rhs: v(),
            },
            OpCode::Cmp {
                kind: CmpKind::Eq,
                result: v(),
                lhs: v(),
                rhs: v(),
            },
            OpCode::MulConst {
                result: v(),
                const_val: v(),
                var: v(),
            },
            OpCode::Cast {
                result: v(),
                value: v(),
                target: CastTarget::Field,
            },
            OpCode::SExt {
                result: v(),
                value: v(),
                from_bits: 8,
                to_bits: 32,
            },
            OpCode::BitRange {
                result: v(),
                value: v(),
                offset: 0,
                width: 8,
            },
            OpCode::Not {
                result: v(),
                value: v(),
            },
            OpCode::Select {
                result: v(),
                cond: v(),
                if_t: v(),
                if_f: v(),
            },
        ];
        for instr in &foldable {
            assert!(instr.is_pure_scalar_fold(), "{instr:?} should be foldable");
            assert_eq!(
                instr.is_pure_scalar_fold(),
                instr.scalar_fold().is_some(),
                "is_pure_scalar_fold must equal scalar_fold().is_some() for {instr:?}"
            );
            // Every foldable op except a witness cast is value-numbered.
            assert!(
                pure_op_operands(instr).is_some(),
                "{instr:?} should be value-numbered"
            );
        }

        // The deliberate asymmetry: a witness cast IS a scalar fold but is NOT value-numbered.
        let witness_cast = OpCode::Cast {
            result: v(),
            value: v(),
            target: CastTarget::WitnessOf,
        };
        assert!(witness_cast.is_pure_scalar_fold());
        assert!(matches!(
            witness_cast.scalar_fold(),
            Some(ScalarFold::Cast { .. })
        ));
        assert!(pure_op_operands(&witness_cast).is_none());

        // The other deliberate asymmetry: pure sequence ops ARE value-numbered but are NOT
        // scalar-foldable (their aggregate results never enter the constant lattice).
        let elem = Type::field();
        let sequence_ops: Vec<OpCode> = vec![
            OpCode::ArrayGet {
                result: v(),
                array: v(),
                index: v(),
            },
            OpCode::ArraySet {
                result: v(),
                array: v(),
                index: v(),
                value: v(),
            },
            OpCode::SliceLen {
                result: v(),
                slice: v(),
            },
            OpCode::MkSeq {
                result: v(),
                elems: vec![v(), v()],
                seq_type: SequenceTargetType::Array(2),
                elem_type: elem.clone(),
            },
            OpCode::MkRepeated {
                result: v(),
                element: v(),
                seq_type: SequenceTargetType::Slice,
                count: 3,
                elem_type: elem.clone(),
            },
            OpCode::MkSeqOfBlob {
                result: v(),
                element_type: elem.clone(),
                blob: v(),
            },
            OpCode::SlicePush {
                dir: SliceOpDir::Back,
                result: v(),
                slice: v(),
                values: vec![v()],
            },
        ];
        for instr in &sequence_ops {
            assert!(
                !instr.is_pure_scalar_fold(),
                "{instr:?} should not be scalar-foldable"
            );
            assert!(instr.scalar_fold().is_none());
            assert!(
                pure_op_operands(instr).is_some(),
                "{instr:?} should be value-numbered"
            );
        }

        // Representative non-folds: not foldable, not value-numbered, no `ScalarFold`.
        let non_folds: Vec<OpCode> = vec![
            OpCode::Assert { value: v() },
            OpCode::Store {
                ptr: v(),
                value: v(),
            },
            OpCode::Load {
                result: v(),
                ptr: v(),
            },
            OpCode::Call {
                results: vec![v()],
                function: CallTarget::Static(main),
                args: vec![v()],
                unconstrained: false,
            },
        ];
        for instr in &non_folds {
            assert!(
                !instr.is_pure_scalar_fold(),
                "{instr:?} should not be foldable"
            );
            assert!(instr.scalar_fold().is_none());
            assert!(
                pure_op_operands(instr).is_none(),
                "{instr:?} should not be value-numbered"
            );
        }
    }

    /// Two values that fold to the *same* constant are congruent, even when computed differently —
    /// the const → congruence coupling.
    #[test]
    fn equal_constants_are_congruent() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c2 = ssa.add_const(Constant::U(32, 2));
        let c3 = ssa.add_const(Constant::U(32, 3));
        let c4 = ssa.add_const(Constant::U(32, 4));
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: c2,
            rhs: c3,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: c1,
            rhs: c4,
        });
        entry.set_terminator(Terminator::Return(vec![a, b]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // 2+3 and 1+4 both equal 5.
        assert!(cc.known_equal(fid, a, b));
        // But not congruent to a different constant.
        assert!(!cc.known_equal(fid, a, c2));
    }

    /// The same expression over the same operands is congruent, commutatively; different operators
    /// or operands are not.
    #[test]
    fn structural_congruence_is_commutative() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
        let (a, b, c, d) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(y, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: c,
            lhs: y,
            rhs: x,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Sub,
            result: d,
            lhs: x,
            rhs: y,
        });
        entry.set_terminator(Terminator::Return(vec![a, b, c, d]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, a, b));
        assert!(cc.known_equal(fid, a, c)); // x+y ≡ y+x
        assert!(!cc.known_equal(fid, a, d)); // x+y ≢ x-y
        assert!(!cc.known_equal(fid, x, y)); // distinct opaque values

        let mut expected = vec![a, b, c];
        expected.sort();
        assert_eq!(cc.congruence_class(fid, a), expected);
    }

    /// Pure sequence ops are value-numbered: two `ArrayGet`s of the same array at the same index
    /// are congruent (a different index is not), `MkSeq`s congruent elementwise are congruent, and
    /// the sequence shape (`seq_type`, `elem_type`) carried in the key keeps differently-shaped
    /// `MkSeq`s apart.
    #[test]
    fn array_ops_are_value_numbered() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (arr, i, j) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let (g1, g2, g3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let (s1, s2, s3, s4) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut()
            .push_parameter(arr, Type::field().array_of(4));
        f.get_entry_mut().push_parameter(i, Type::u(32));
        f.get_entry_mut().push_parameter(j, Type::u(32));
        let entry = f.get_entry_mut();
        // g1, g2 are arr[i]; g3 is arr[j].
        entry.push_instruction(OpCode::ArrayGet {
            result: g1,
            array: arr,
            index: i,
        });
        entry.push_instruction(OpCode::ArrayGet {
            result: g2,
            array: arr,
            index: i,
        });
        entry.push_instruction(OpCode::ArrayGet {
            result: g3,
            array: arr,
            index: j,
        });
        // s1 and s2 are the same array `[arr[i], arr[j]]` built twice — congruent elementwise.
        entry.push_instruction(OpCode::MkSeq {
            result: s1,
            elems: vec![g1, g3],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::field(),
        });
        entry.push_instruction(OpCode::MkSeq {
            result: s2,
            elems: vec![g2, g3],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::field(),
        });
        // s3 differs only in sequence kind (Slice), s4 only in element type (u32): neither merges.
        entry.push_instruction(OpCode::MkSeq {
            result: s3,
            elems: vec![g1, g3],
            seq_type: SequenceTargetType::Slice,
            elem_type: Type::field(),
        });
        entry.push_instruction(OpCode::MkSeq {
            result: s4,
            elems: vec![g1, g3],
            seq_type: SequenceTargetType::Array(2),
            elem_type: Type::u(32),
        });
        entry.set_terminator(Terminator::Return(vec![g1, g2, g3, s1, s2, s3, s4]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, g1, g2)); // arr[i] ≡ arr[i]
        assert!(!cc.known_equal(fid, g1, g3)); // arr[i] ≢ arr[j]
        assert!(cc.known_equal(fid, s1, s2)); // elementwise-congruent arrays
        assert!(!cc.known_equal(fid, s1, s3)); // different seq_type
        assert!(!cc.known_equal(fid, s1, s4)); // different elem_type
    }

    /// φ-operands come from *executable* edges only: a parameter whose value differs solely on a
    /// dead in-edge is still congruent to one that agrees on the live edge.
    #[test]
    fn phi_congruence_excludes_dead_edges() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (x, y, p, q) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let live = f.add_block();
        let dead = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(c_true, live, dead));
        // Live edge agrees on both params; the dead edge would have forced them apart.
        f.get_block_mut(live)
            .set_terminator(Terminator::Jmp(merge, vec![x, x]));
        f.get_block_mut(dead)
            .set_terminator(Terminator::Jmp(merge, vec![x, y]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(p, Type::field());
        merge_block.push_parameter(q, Type::field());
        merge_block.set_terminator(Terminator::Return(vec![p, q]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, p, q));
    }

    /// The same merge with *both* edges live keeps the parameters apart — congruence is genuinely
    /// reachability-sensitive.
    #[test]
    fn phi_distinguished_when_both_edges_live() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, x, y, p, q) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let e1 = f.add_block();
        let e2 = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, e1, e2));
        f.get_block_mut(e1)
            .set_terminator(Terminator::Jmp(merge, vec![x, x]));
        f.get_block_mut(e2)
            .set_terminator(Terminator::Jmp(merge, vec![x, y]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(p, Type::field());
        merge_block.push_parameter(q, Type::field());
        merge_block.set_terminator(Terminator::Return(vec![p, q]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(!cc.known_equal(fid, p, q));
    }

    /// The optimistic win pessimistic value numbering cannot reach: two parallel induction variables
    /// that start equal and step identically are congruent across the loop back-edge.
    #[test]
    fn loop_carried_parallel_induction_is_congruent() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c0 = ssa.add_const(Constant::U(32, 0));
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c10 = ssa.add_const(Constant::U(32, 10));
        let (i, j, lt, i2, j2) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let header = f.add_block();
        let body = f.add_block();
        let exit = f.add_block();

        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(header, vec![c0, c0]));
        let header_block = f.get_block_mut(header);
        header_block.push_parameter(i, Type::u(32));
        header_block.push_parameter(j, Type::u(32));
        header_block.push_instruction(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i,
            rhs: c10,
        });
        header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
        let body_block = f.get_block_mut(body);
        body_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: i2,
            lhs: i,
            rhs: c1,
        });
        body_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: j2,
            lhs: j,
            rhs: c1,
        });
        body_block.set_terminator(Terminator::Jmp(header, vec![i2, j2]));
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![i, j]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, i, j)); // loop-carried congruence
        assert!(cc.known_equal(fid, i2, j2)); // and their parallel updates

        // The dominance-aware leader is the member at the dominating definition: the two header
        // parameters share a definition site (block entry), so the lower-id one leads; in the body,
        // the earlier `i2` leads the later `j2`.
        assert_eq!(cc.leader(fid, i), Some(i));
        assert_eq!(cc.leader(fid, j), Some(i));
        assert_eq!(cc.leader(fid, i2), Some(i2));
        assert_eq!(cc.leader(fid, j2), Some(i2));
    }

    /// A congruent definition that dominates another is its leader (a legal redirect target); the
    /// dominated occurrence redirects to the dominating one, and the dominating one leads itself.
    #[test]
    fn leader_is_dominating_definition() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y, a, b) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let next = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        // `a = x + y` in the entry, recomputed as `b = x + y` in a strictly dominated block.
        f.get_entry_mut().push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(next, vec![]));
        let next_block = f.get_block_mut(next);
        next_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        });
        next_block.set_terminator(Terminator::Return(vec![a, b]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, a, b));
        assert_eq!(cc.leader(fid, a), Some(a)); // dominating definition leads itself
        assert_eq!(cc.leader(fid, b), Some(a)); // dominated occurrence redirects to it
    }

    /// Two congruent ops in one block: the earlier one (by instruction index) leads the later.
    #[test]
    fn leader_is_earlier_in_block() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y, a, b) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        });
        entry.set_terminator(Terminator::Return(vec![a, b]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, a, b));
        assert_eq!(cc.leader(fid, a), Some(a));
        assert_eq!(cc.leader(fid, b), Some(a));
    }

    /// Congruent occurrences in two incomparable branches (and one at the merge) have no single
    /// dominating member, so each is its own leader — the leader is never a non-dominating member,
    /// so no illegal cross-branch redirect is offered.
    #[test]
    fn leader_never_crosses_incomparable_branches() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, x, y, a, b, c) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let e1 = f.add_block();
        let e2 = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, e1, e2));
        // `x + y` recomputed in both incomparable branches and again at the merge.
        let e1_block = f.get_block_mut(e1);
        e1_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        });
        e1_block.set_terminator(Terminator::Jmp(merge, vec![]));
        let e2_block = f.get_block_mut(e2);
        e2_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        });
        e2_block.set_terminator(Terminator::Jmp(merge, vec![]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: c,
            lhs: x,
            rhs: y,
        });
        merge_block.set_terminator(Terminator::Return(vec![a, b, c]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // All three are congruent...
        assert!(cc.known_equal(fid, a, b));
        assert!(cc.known_equal(fid, a, c));
        // ...but none dominates another, so each leads itself (no cross-branch redirect).
        assert_eq!(cc.leader(fid, a), Some(a));
        assert_eq!(cc.leader(fid, b), Some(b));
        assert_eq!(cc.leader(fid, c), Some(c));
    }

    /// A computed value proven constant is congruent to the interned constant of the same value, and
    /// that constant — available throughout the function — is its leader.
    #[test]
    fn leader_of_constant_class_is_the_interned_constant() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c2 = ssa.add_const(Constant::U(32, 2));
        let c3 = ssa.add_const(Constant::U(32, 3));
        let c5 = ssa.add_const(Constant::U(32, 5));
        let a = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        // `a = 2 + 3` folds to 5, congruent to the interned constant `c5`.
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: c2,
            rhs: c3,
        });
        entry.set_terminator(Terminator::Return(vec![a, c5]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, a, c5));
        assert_eq!(cc.leader(fid, a), Some(c5)); // the constant dominates everywhere
        assert_eq!(cc.leader(fid, c5), Some(c5));
    }

    /// Assert-vacuum soundness: `assert(x == 5)` pins `x` to 5 *conditionally* in dominated blocks,
    /// but never unconditionally — so a global fold (SCCP) can't fold `x` and vacuum the assert.
    #[test]
    fn assert_eq_const_is_conditional_not_unconditional() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::U(32, 5));
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);

        // Unconditionally `x` is unknown — the assert never enters the global lattice.
        assert_eq!(cc.const_of(fid, x), None);
        assert!(cc.new_const_values(fid).iter().all(|(v, _)| *v != x));
        // Conditionally `x == 5` at every block the assert *strictly* dominates ...
        assert_eq!(
            cc.asserted_const(fid, after, x).as_deref(),
            Some(&Constant::U(32, 5))
        );
        // ... but not in the asserting block itself (block-entry granularity).
        assert_eq!(cc.asserted_const(fid, entry_id, x), None);
    }

    /// `assert(b)` proves `b == true` conditionally in dominated blocks, never unconditionally.
    #[test]
    fn assert_bool_is_conditional() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let b = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(b, Type::u(1));
        f.get_entry_mut()
            .push_instruction(OpCode::Assert { value: b });
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert_eq!(cc.const_of(fid, b), None);
        assert_eq!(
            cc.asserted_const(fid, after, b).as_deref(),
            Some(&Constant::U(1, 1))
        );
    }

    /// `assert(x == y)` with neither side constant records a *conditional* equality — distinct from
    /// structural congruence, which (unconditionally) keeps the two parameters apart.
    #[test]
    fn assert_eq_pure_equality_is_conditional() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(y, Type::u(32));
        f.get_entry_mut().push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.asserted_equal(fid, after, x, y));
        assert!(cc.asserted_equal(fid, after, y, x)); // symmetric
        assert!(!cc.asserted_equal(fid, entry_id, x, y)); // not in the asserting block
        // Structural congruence (unconditional) does not see the assert.
        assert!(!cc.known_equal(fid, x, y));
    }

    /// The false edge of `if x == y` proves `x != y` at its target and the blocks that target
    /// dominates — and nowhere else.
    #[test]
    fn disequality_from_false_edge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(y, Type::u(32));
        f.get_entry_mut().push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: x,
            rhs: y,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // x != y on the false-edge target and blocks it dominates.
        assert!(cc.known_unequal(fid, else_b, x, y));
        assert!(cc.known_unequal(fid, else_b, y, x)); // symmetric
        assert!(cc.known_unequal(fid, after, x, y));
        // Not on the true edge, nor before the branch.
        assert!(!cc.known_unequal(fid, then_b, x, y));
        assert!(!cc.known_unequal(fid, entry_id, x, y));
    }

    /// An unpinned witness write forwards `r → v`; distinct witnesses get distinct forwards and are
    /// never congruent; a *pinned* write (a real constraint) forwards nothing.
    #[test]
    fn unpinned_witness_forwards_distinct_not_merged() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (v1, v2) = (ssa.fresh_value(), ssa.fresh_value());
        let (r1, r2, rp) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(v1, Type::u(32));
        f.get_entry_mut().push_parameter(v2, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r1),
            value: v1,
            pinned: false,
        });
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r2),
            value: v2,
            pinned: false,
        });
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(rp),
            value: v1,
            pinned: true,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert_eq!(cc.witness_forward(fid, r1), [v1].as_slice());
        assert_eq!(cc.witness_forward(fid, r2), [v2].as_slice());
        // A pinned write carries a real `r == v` constraint — it is not a free witness, no forward.
        assert!(cc.witness_forward(fid, rp).is_empty());
        // Two distinct free witnesses are never unified (the hard non-merge prohibition).
        assert!(!cc.known_equal(fid, r1, r2));
    }

    /// `witness_forward` is the sorted union of *both* readings of the one witness↔value
    /// correspondence: the `WriteWitness` hint (`r → v`) and every `value_of(r)` read (`r → w`).
    /// Distinct witnesses keep disjoint sets — no cross-witness union.
    #[test]
    fn witness_forward_unions_hint_and_value_of_reads() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (v, v2) = (ssa.fresh_value(), ssa.fresh_value()); // honest hint payloads
        let (r, r2) = (ssa.fresh_value(), ssa.fresh_value()); // witness handles
        let (w1, w2) = (ssa.fresh_value(), ssa.fresh_value()); // two distinct `value_of(r)` reads

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(v, Type::u(32));
        f.get_entry_mut().push_parameter(v2, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r),
            value: v,
            pinned: false,
        });
        // `value_of(r)` strips r's witness wrapper: w1, w2 are honestly equal to r. Two separate
        // reads stay distinct (ValueOf is excluded from value-numbering), so both join r's set.
        entry.push_instruction(OpCode::Cast {
            result: w1,
            value: r,
            target: CastTarget::ValueOf,
        });
        entry.push_instruction(OpCode::Cast {
            result: w2,
            value: r,
            target: CastTarget::ValueOf,
        });
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r2),
            value: v2,
            pinned: false,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);

        // r's honest-value set is the sorted union of the hint and both `value_of` reads.
        let mut expected = vec![v, w1, w2];
        expected.sort_unstable();
        assert_eq!(cc.witness_forward(fid, r), expected.as_slice());

        // The unrelated witness keeps a disjoint, single-element set — no cross-witness union.
        assert_eq!(cc.witness_forward(fid, r2), [v2].as_slice());
        assert!(!cc.known_equal(fid, r, r2));
    }

    /// An assert placed in the *last* block proves nothing by dominance (nothing follows it), but it
    /// post-dominates everything upstream — so on accepting runs its fact holds at those earlier
    /// blocks. The asserted value (`x`, a parameter) is in scope throughout.
    #[test]
    fn assert_below_use_holds_via_post_dominance() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::U(32, 5));
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let mid = f.add_block();
        let tail = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(mid, vec![]));
        f.get_block_mut(mid)
            .set_terminator(Terminator::Jmp(tail, vec![]));
        f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        });
        f.get_block_mut(tail)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // `tail` post-dominates `mid` and `entry`, so `x == 5` holds at both — a fact pure dominance
        // (the assert is last) would miss entirely.
        assert_eq!(
            cc.asserted_const(fid, mid, x).as_deref(),
            Some(&Constant::U(32, 5))
        );
        assert_eq!(
            cc.asserted_const(fid, entry_id, x).as_deref(),
            Some(&Constant::U(32, 5))
        );
        // Still never recorded at the asserting block's own entry (block-entry granularity).
        assert_eq!(cc.asserted_const(fid, tail, x), None);
    }

    /// Post-dominance fan-out is gated on the asserted value being in scope: a value defined *below*
    /// the target block must not have the fact attributed to that target (the guard dominance
    /// otherwise supplies for free).
    #[test]
    fn post_dominance_respects_value_scope() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c10 = ssa.add_const(Constant::U(32, 10));
        let p = ssa.fresh_value();
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let mid = f.add_block();
        let tail = f.add_block();
        f.get_entry_mut().push_parameter(p, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(mid, vec![]));
        // `x` is defined in `mid`, *below* `entry`.
        f.get_block_mut(mid)
            .push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: x,
                lhs: p,
                rhs: p,
            });
        f.get_block_mut(mid)
            .set_terminator(Terminator::Jmp(tail, vec![]));
        f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c10,
        });
        f.get_block_mut(tail)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // At `mid`, `x` is in scope and `tail` post-dominates it ⇒ the fact holds.
        assert_eq!(
            cc.asserted_const(fid, mid, x).as_deref(),
            Some(&Constant::U(32, 10))
        );
        // At `entry`, `x` is not yet defined — the in-scope guard withholds the fact even though
        // `tail` post-dominates `entry`.
        assert_eq!(cc.asserted_const(fid, entry_id, x), None);
    }

    /// An assert on only one arm of a branch neither dominates nor post-dominates the blocks around
    /// the branch, so its fact must not leak there — the other arm reaches `exit` without it.
    #[test]
    fn assert_on_one_branch_does_not_post_dominate() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::U(32, 5));
        let cond = ssa.fresh_value();
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        // The assert lives only on the `then` branch.
        f.get_block_mut(then_b).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        });
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        f.get_block_mut(merge)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // `x` is in scope everywhere, so only the missing dominance/post-dominance keeps the fact
        // out of `entry` and `merge` (the `else` path skips the assert).
        assert_eq!(cc.asserted_const(fid, entry_id, x), None);
        assert_eq!(cc.asserted_const(fid, merge, x), None);
        // And never at the asserting block's own entry.
        assert_eq!(cc.asserted_const(fid, then_b, x), None);
    }

    /// A post-dominating *pure* equality (`assert(x == y)`, neither side constant) requires *both*
    /// sides in scope at the target.
    #[test]
    fn post_dominating_assert_eq_needs_both_sides_in_scope() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let x = ssa.fresh_value();
        let y = ssa.fresh_value();
        let p = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let mid = f.add_block();
        let tail = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(p, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(mid, vec![]));
        // `y` is defined in `mid`, below `entry`.
        f.get_block_mut(mid)
            .push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: y,
                lhs: p,
                rhs: p,
            });
        f.get_block_mut(mid)
            .set_terminator(Terminator::Jmp(tail, vec![]));
        f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        });
        f.get_block_mut(tail)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // Both sides live at `mid` ⇒ the post-dominating equality holds, symmetrically.
        assert!(cc.asserted_equal(fid, mid, x, y));
        assert!(cc.asserted_equal(fid, mid, y, x));
        // `y` is undefined at `entry`, so the equality is withheld there.
        assert!(!cc.asserted_equal(fid, entry_id, x, y));
    }

    /// A callee that returns a constant: the call result folds interprocedurally in the caller's
    /// context, while the SCCP-visible intraprocedural view leaves it `Bottom`.
    #[test]
    fn interproc_constant_return_folds_at_call_site() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let helper = ssa.add_function("helper".to_string());
        let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
        let r = ssa.fresh_value();

        {
            let hf = ssa.get_function_mut(helper);
            hf.add_return_type(Type::field());
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![c5]));
        }
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            let entry = mf.get_entry_mut();
            entry.push_instruction(OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(helper),
                args: vec![],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r]));
        }

        let cc = run_in_test(&ssa);
        assert_eq!(
            cc.const_of_in(main_id, &Context::empty(), r).as_deref(),
            Some(&Constant::Field(Field::from(5u64)))
        );
        // The intraprocedural view (what SCCP reads) never folds a call result.
        assert_eq!(cc.const_of(main_id, r), None);
    }

    /// A pass-through callee (returns its parameter): the result takes the argument's constant at
    /// the call site, and the callee's parameter is seeded to it per context.
    #[test]
    fn interproc_passthrough_seeds_param_and_result() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let id = ssa.add_function("id".to_string());
        let p = ssa.fresh_value();
        let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
        let r = ssa.fresh_value();

        {
            let hf = ssa.get_function_mut(id);
            hf.add_return_type(Type::field());
            hf.get_entry_mut().push_parameter(p, Type::field());
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![p]));
        }
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            let entry = mf.get_entry_mut();
            entry.push_instruction(OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(id),
                args: vec![c7],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r]));
        }

        let cc = run_in_test(&ssa);
        assert_eq!(
            cc.const_of_in(main_id, &Context::empty(), r).as_deref(),
            Some(&Constant::Field(Field::from(7u64)))
        );
        let ctxs = cc.contexts_of(id);
        assert_eq!(ctxs.len(), 1);
        assert_eq!(
            cc.const_of_in(id, &ctxs[0], p).as_deref(),
            Some(&Constant::Field(Field::from(7u64)))
        );
    }

    /// The context-parameterized conditional queries. The branch-fact family
    /// (`const_in_block_in`) is context-*precise* — it sees the per-context parameter constant the
    /// intraprocedural query cannot — while the assert/witness family forwards to the (sound,
    /// context-independent) intraprocedural facts.
    #[test]
    fn context_parameterized_conditional_queries() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let id = ssa.add_function("id".to_string());
        let p = ssa.fresh_value();
        let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
        let r = ssa.fresh_value();

        {
            let hf = ssa.get_function_mut(id);
            hf.add_return_type(Type::field());
            hf.get_entry_mut().push_parameter(p, Type::field());
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![p]));
        }
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            let entry = mf.get_entry_mut();
            entry.push_instruction(OpCode::Call {
                results: vec![r],
                function: CallTarget::Static(id),
                args: vec![c7],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r]));
        }

        let entry_id = ssa.get_function(id).get_entry_id();
        let cc = run_in_test(&ssa);
        let ctxs = cc.contexts_of(id);
        assert_eq!(ctxs.len(), 1);
        let ctx = &ctxs[0];

        // Intraprocedurally `p` is unconstrained; under the single call context it is the constant 7.
        assert_eq!(cc.const_in_block(id, entry_id, p), None);
        assert_eq!(
            cc.const_in_block_in(id, ctx, entry_id, p).as_deref(),
            Some(&Constant::Field(Field::from(7u64)))
        );
    }

    /// Two call sites of one helper get distinct contexts with distinct per-context parameter
    /// constants — the 1-CFA win.
    #[test]
    fn interproc_two_call_sites_are_distinguished() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let id = ssa.add_function("id".to_string());
        let p = ssa.fresh_value();
        let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
        let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
        let (r5, r7) = (ssa.fresh_value(), ssa.fresh_value());

        {
            let hf = ssa.get_function_mut(id);
            hf.add_return_type(Type::field());
            hf.get_entry_mut().push_parameter(p, Type::field());
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![p]));
        }
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            let entry = mf.get_entry_mut();
            entry.push_instruction(OpCode::Call {
                results: vec![r5],
                function: CallTarget::Static(id),
                args: vec![c5],
                unconstrained: false,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r7],
                function: CallTarget::Static(id),
                args: vec![c7],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r5, r7]));
        }

        let cc = run_in_test(&ssa);
        assert_eq!(
            cc.const_of_in(main_id, &Context::empty(), r5).as_deref(),
            Some(&Constant::Field(Field::from(5u64)))
        );
        assert_eq!(
            cc.const_of_in(main_id, &Context::empty(), r7).as_deref(),
            Some(&Constant::Field(Field::from(7u64)))
        );
        let ctxs = cc.contexts_of(id);
        assert_eq!(ctxs.len(), 2, "two call sites → two contexts");
        let seen: Vec<Constant> = ctxs
            .iter()
            .filter_map(|ctx| cc.const_of_in(id, ctx, p).map(|c| (*c).clone()))
            .collect();
        assert_eq!(seen.len(), 2);
        assert!(seen.contains(&Constant::Field(Field::from(5u64))));
        assert!(seen.contains(&Constant::Field(Field::from(7u64))));
    }

    /// An unconstrained call's result is advice, not circuit-constrained: it never folds (even when
    /// the callee returns a constant) and stays an opaque singleton.
    #[test]
    fn unconstrained_call_result_is_opaque() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let helper = ssa.add_function("helper".to_string());
        let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
        let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());

        {
            let hf = ssa.get_function_mut(helper);
            hf.add_return_type(Type::field());
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![c5]));
        }
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            mf.add_return_type(Type::field());
            let entry = mf.get_entry_mut();
            entry.push_instruction(OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(helper),
                args: vec![],
                unconstrained: true,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(helper),
                args: vec![],
                unconstrained: true,
            });
            entry.set_terminator(Terminator::Return(vec![r1, r2]));
        }

        let cc = run_in_test(&ssa);
        // Even interprocedurally, an unconstrained result is never folded ...
        assert_eq!(cc.const_of_in(main_id, &Context::empty(), r1), None);
        assert_eq!(cc.const_of(main_id, r1), None);
        // ... and two such results are not merged.
        assert!(!cc.known_equal(main_id, r1, r2));
    }

    /// Cross-call congruence: a callee whose return is a deterministic function of its arguments is
    /// value-numbered cross-call, so two calls with congruent arguments yield congruent results —
    /// and a call with a non-congruent argument does not.
    #[test]
    fn cross_call_congruence_for_deterministic_callee() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let dbl = ssa.add_function("dbl".to_string());
        let p = ssa.fresh_value();
        let psum = ssa.fresh_value();

        // dbl(p) = p + p — deterministic in its argument.
        {
            let hf = ssa.get_function_mut(dbl);
            hf.add_return_type(Type::field());
            hf.get_entry_mut().push_parameter(p, Type::field());
            hf.get_entry_mut().push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: psum,
                lhs: p,
                rhs: p,
            });
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![psum]));
        }

        let c1 = ssa.add_const(Constant::Field(Field::from(1u64)));
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
        let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            mf.add_return_type(Type::field());
            mf.add_return_type(Type::field());
            mf.get_entry_mut().push_parameter(x, Type::field());
            mf.get_entry_mut().push_parameter(y, Type::field());
            let entry = mf.get_entry_mut();
            // `a` and `b` are structurally congruent (both `x + 1`); `y` is distinct.
            entry.push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: a,
                lhs: x,
                rhs: c1,
            });
            entry.push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: b,
                lhs: x,
                rhs: c1,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(dbl),
                args: vec![a],
                unconstrained: false,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(dbl),
                args: vec![b],
                unconstrained: false,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r3],
                function: CallTarget::Static(dbl),
                args: vec![y],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
        }

        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(main_id, r1, r2)); // congruent args ⇒ congruent results
        assert!(!cc.known_equal(main_id, r1, r3)); // distinct arg ⇒ distinct result
    }

    /// A callee whose return carries a fresh witness is *not* a deterministic function of its
    /// arguments, so its results are never numbered cross-call — two such calls stay distinct (the
    /// determinism gate that protects the free-witness non-merge prohibition).
    #[test]
    fn cross_call_no_congruence_for_nondeterministic_callee() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let rnd = ssa.add_function("rnd".to_string());
        let w = ssa.fresh_value();

        {
            let hf = ssa.get_function_mut(rnd);
            hf.add_return_type(Type::field());
            hf.get_entry_mut().push_instruction(OpCode::FreshWitness {
                result: w,
                result_type: Type::field(),
            });
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![w]));
        }
        let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            mf.add_return_type(Type::field());
            let entry = mf.get_entry_mut();
            entry.push_instruction(OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(rnd),
                args: vec![],
                unconstrained: false,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(rnd),
                args: vec![],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r1, r2]));
        }

        let cc = run_in_test(&ssa);
        assert!(!cc.known_equal(main_id, r1, r2));
    }

    /// Determinism is interprocedural: `outer` is deterministic because the `inner` it calls is, so
    /// its results are numbered cross-call too.
    #[test]
    fn cross_call_congruence_is_transitive() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let inner = ssa.add_function("inner".to_string());
        let outer = ssa.add_function("outer".to_string());

        let ip = ssa.fresh_value();
        let isum = ssa.fresh_value();
        {
            let hf = ssa.get_function_mut(inner);
            hf.add_return_type(Type::field());
            hf.get_entry_mut().push_parameter(ip, Type::field());
            hf.get_entry_mut().push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: isum,
                lhs: ip,
                rhs: ip,
            });
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![isum]));
        }
        let op = ssa.fresh_value();
        let ores = ssa.fresh_value();
        {
            let hf = ssa.get_function_mut(outer);
            hf.add_return_type(Type::field());
            hf.get_entry_mut().push_parameter(op, Type::field());
            hf.get_entry_mut().push_instruction(OpCode::Call {
                results: vec![ores],
                function: CallTarget::Static(inner),
                args: vec![op],
                unconstrained: false,
            });
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![ores]));
        }

        let c1 = ssa.add_const(Constant::Field(Field::from(1u64)));
        let x = ssa.fresh_value();
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());
        let (r1, r2) = (ssa.fresh_value(), ssa.fresh_value());
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            mf.add_return_type(Type::field());
            mf.get_entry_mut().push_parameter(x, Type::field());
            let entry = mf.get_entry_mut();
            entry.push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: a,
                lhs: x,
                rhs: c1,
            });
            entry.push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: b,
                lhs: x,
                rhs: c1,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(outer),
                args: vec![a],
                unconstrained: false,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(outer),
                args: vec![b],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r1, r2]));
        }

        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(main_id, r1, r2));
    }

    /// A callee that returns a pure transform of an array parameter (`p[0]`) is a deterministic
    /// function of its argument now that sequence ops are value-numbered, so two calls with
    /// congruent array arguments yield congruent results. Before this change the `ArrayGet` tainted
    /// the return as non-deterministic and the calls stayed distinct.
    #[test]
    fn cross_call_congruence_for_array_returning_callee() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let pick = ssa.add_function("pick".to_string());
        let p = ssa.fresh_value();
        let elem = ssa.fresh_value();
        let c0 = ssa.add_const(Constant::U(32, 0));

        // pick(p) = p[0] — deterministic in its array argument.
        {
            let hf = ssa.get_function_mut(pick);
            hf.add_return_type(Type::field());
            hf.get_entry_mut()
                .push_parameter(p, Type::field().array_of(1));
            hf.get_entry_mut().push_instruction(OpCode::ArrayGet {
                result: elem,
                array: p,
                index: c0,
            });
            hf.get_entry_mut()
                .set_terminator(Terminator::Return(vec![elem]));
        }

        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
        let (a, b, d) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let (r1, r2, r3) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        {
            let mf = ssa.get_function_mut(main_id);
            mf.add_return_type(Type::field());
            mf.add_return_type(Type::field());
            mf.add_return_type(Type::field());
            mf.get_entry_mut().push_parameter(x, Type::field());
            mf.get_entry_mut().push_parameter(y, Type::field());
            let entry = mf.get_entry_mut();
            // `a` and `b` are congruent single-element arrays (`[x]`); `d` is `[y]`.
            entry.push_instruction(OpCode::MkSeq {
                result: a,
                elems: vec![x],
                seq_type: SequenceTargetType::Array(1),
                elem_type: Type::field(),
            });
            entry.push_instruction(OpCode::MkSeq {
                result: b,
                elems: vec![x],
                seq_type: SequenceTargetType::Array(1),
                elem_type: Type::field(),
            });
            entry.push_instruction(OpCode::MkSeq {
                result: d,
                elems: vec![y],
                seq_type: SequenceTargetType::Array(1),
                elem_type: Type::field(),
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r1],
                function: CallTarget::Static(pick),
                args: vec![a],
                unconstrained: false,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r2],
                function: CallTarget::Static(pick),
                args: vec![b],
                unconstrained: false,
            });
            entry.push_instruction(OpCode::Call {
                results: vec![r3],
                function: CallTarget::Static(pick),
                args: vec![d],
                unconstrained: false,
            });
            entry.set_terminator(Terminator::Return(vec![r1, r2, r3]));
        }

        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(main_id, r1, r2)); // congruent array args ⇒ congruent results
        assert!(!cc.known_equal(main_id, r1, r3)); // distinct array arg ⇒ distinct result
    }
}
