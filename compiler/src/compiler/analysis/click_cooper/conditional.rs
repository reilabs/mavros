//! Facts that are downstream of conditional expressions (asserts, equalities, disequalities, and
//! unpinned witness forwarding).
//!
//! These facts only hold on _accepting_ runs because a constraint establishes them. Per the
//! module-level soundness contract they are sound only under _constraint-preserving_ use (local
//! rewrites that maintain the establishing constraint). The _Soundness on Rejecting Runs_ section
//! of [`super`] states the observation model under which divergence on non-accepting runs is
//! unobservable. The computation here is performed post-convergence using a single pass over the
//! already solved function facts structure, plus the dominance and post-dominance relations of the
//! CFG.
//!
//! # Dominance, not Dataflow
//!
//! An assert is a _global_ constraint, not a runtime check: on an accepting run it holds at every
//! program point. So an assert in reachable block `B` contributes its fact to the entry of every
//! reachable block `C` that `B` strictly dominates — the assert has _already run_ before `C`, so
//! the constraint is established there. Static dominance is sound because every statically-dominating
//! block also dominates on the executable subgraph (pruning paths only adds dominators), a
//! conservative under-approximation of executable-path dominance, and thus subsumes the
//! intersect-at-join dataflow used by branch facts. Because `def(v) dom B dom C`, the asserted value
//! `v` is transitively live at `C`, so the dominance direction needs no separate in-scope check.
//!
//! # Anticipation (Post-Dominance)
//!
//! The mirror direction: an assert in `B` is _bound to run_ at every reachable block `C` that `B`
//! strictly post-dominates. A run visiting `C` either reaches the assert, which checks the
//! **genuine** operand values, aborts on an earlier constraint, or hangs. Every non-reaching
//! outcome rejects (a non-terminating run produces no witness), so on all accepting runs the fact
//! already holds at `C`. These are the `anticipated_*` channels, disjoint from the `asserted_*`
//! ones, and they need two gates the dominance direction gets for free:
//!
//! - **Gate 1, Value Stability:** A ValueId defined inside a CFG cycle may denote a _different_
//!   value on each encounter, so the binding the bound-to-run assert checks need not be the binding
//!   at the earlier point (`assert v == 10` after a loop says nothing about `v` mid-loop). Each
//!   value in an anticipated fact must pass one of two stability conditions at the target block
//!   `C`: it is _single-valued per invocation_ ([`ValueStability`]), or its binding is already
//!   _final at `C`_ ([`BlockReach`]). For an equality pair each member is gated independently,
//!   possibly by different conditions.
//! - **Gate 2, SSA scope:** `B post-dom C` does not imply the fact's values are defined at `C`,
//!   so each value's defining block must dominate `C` (reflexively; a member defined by an
//!   instruction of `C` itself is a valid redirect target only at later indices, which the leader
//!   table enforces per leader — see [`ConditionalFacts::anticipated_leaders`]).
//!
//! Post-dominance is measured as the **union** of the static relation and, when this build's solver
//! proved a static edge non-executable, the executable-subgraph relation. Each disjunct is sound on
//! its own: honest runs traverse only executable edges, so both share the reach-abort-or-hang
//! trichotomy above. See [`super::exec_view`] for why neither relation subsumes the other and the
//! union never yields fewer facts than the static relation alone. Disequalities have no anticipated
//! direction at all: a branch-edge fact is path-conditional, not bound to run.
//!
//! Consumers inherit one extra obligation: an anticipated fact is justified by its assert _still
//! checking the genuine values_, so it must never rewrite the operands of any `Assert`/`AssertCmp`.
//! Otherwise two bound-to-run checks of the same fact erase each other — the dominance fact from
//! the first folds the second's operands into a droppable tautology while the anticipated fact from
//! the second folds the first's — and the constraint silently vanishes.
//!
//! The dominance channel needs no such rule: its rewrites flow strictly forward of their
//! establisher, so the earliest check of any fact is never weakened.
//!
//! # Index Granularity Within the Asserting Block
//!
//! The dominance fan-out above is _strict_ (`B != C`), so it never attributes an assert to its own
//! block `B`. A use in `B` _after_ the assert is instead recovered at index granularity: each
//! assert is recorded against `B` together with its instruction index (the `local_*` maps), and a
//! query at a program point in `B` sees the fact only when its use-index is strictly greater than
//! the assert's.
//!
//! Strict `>` is the index-granular form of the "already ran" (dominance) direction, and it
//! self-protects the constraint: an assert never informs its _own_ operands (same index), so it
//! cannot be folded into a vacuous tautology — not even by a later, contradictory assert in `B`
//! (the earliest establisher wins — first-writer-wins). The asserted operands are the assert's own
//! inputs, hence defined before it, hence in scope at every later index — so the `local_*` path
//! needs no `in_scope` check.
//!
//! A use textually _before_ a same-block assert is the index-granular form of the _anticipated_
//! direction, read from the same `local_*` maps with the opposite (`>`) index rule: straight-line
//! execution from the use reaches the assert within the same block iteration, on the same
//! bindings, so neither stability nor scope needs checking. The `q == i` point matches neither
//! rule, so an assert never informs its own operands in either direction, and Gate 3 (see the
//! "Anticipation" section) keeps the anticipated reads out of _other_ asserts' operands. Together
//! these close the contradiction hazard the before-the-assert direction would otherwise open.
//!
//! The anticipated _leader_ table folds both index directions together: since a same-block pair
//! holds at every use-index except its establishing assert's own (`> i` because the assert has
//! already run, `< i` because it is bound to run), the union equality classes carry no per-pair
//! thresholds at all, and the `q == i` self-exclusion rests **wholly on Gate 3** there (every
//! consumer of the anticipated channel is barred from `Assert`/`AssertCmp` operands). The direct
//! const/eq scans above keep their strict index rules, so the self-protection stays structural for
//! them and for the dominance channel.
//!
//! A disequality is an _edge_ fact (the false edge of an equality branch): it is path-conditional —
//! true only on the taken edge — so it holds at the entry of its target block `else_b` only when
//! that edge is the block's sole executable in-edge, and from there propagates to every block
//! `else_b` _dominates_ (reflexively).

use std::sync::Arc;

use crate::{
    collections::{HashMap, HashSet, UnionFind},
    compiler::{
        analysis::{
            click_cooper::{
                def_order::DefOrder,
                exec_view::ExecView,
                lattice::{Constness, bool_constant},
                solver::FunctionFacts,
                stability::{BlockReach, CyclicBlocks, ValueStability},
                summary::{DetSummaries, det_return},
            },
            flow_analysis::CFG,
            types::FunctionTypeInfo,
        },
        ssa::{
            BlockId, ProgramPoint, Terminator, ValueId,
            hlssa::{CastTarget, CmpKind, Constant, HLFunction, HLSSAConstantsSnapshot, OpCode},
        },
    },
};

// CONDITIONAL FACTS
// ================================================================================================

/// The conditional facts of one function.
///
/// The cross-block maps ([`Self::assert_const`], [`Self::assert_eq`], [`Self::diseq`], and their
/// anticipated counterparts [`Self::anticipated_const`], [`Self::anticipated_eq`]) are keyed by
/// the block at whose **entry** the fact holds.
///
/// The two `local_*` maps are keyed by the _asserting_ block and additionally carry the
/// establishing assert's instruction index; the dominance queries read them at use-indices strictly
/// after it, the anticipated queries strictly before (see their field docs and the module "Index
/// Granularity" section).
#[derive(Debug, Default)]
pub(crate) struct ConditionalFacts {
    /// Values forced to a constant by a dominating assert: `Assert{v}` ⇒ `v = true`,
    /// `AssertCmp{Eq, x, c}` (with one side an unconditional _scalar_ constant `c`) ⇒ the other
    /// side `= c`.
    ///
    /// Aggregate (`Blob`) pins are recorded nowhere as Blob constants are never surfaced to
    /// consumers.
    assert_const: HashMap<BlockId, HashMap<ValueId, Arc<Constant>>>,

    /// Equalities established by a dominating `AssertCmp{Eq, a, b}` where neither side is a
    /// (unconditional) constant — a _conditional_ analog of congruence's `known_equal`.
    ///
    /// Each pair is stored canonically as `(min, max)` by value id, so `(a, b)` and `(b, a)` dedup
    /// as one.
    assert_eq: HashMap<BlockId, Vec<(ValueId, ValueId)>>,

    /// Values pinned to a constant by an assert _within_ the keyed block, each paired with the
    /// instruction index of the establishing assert.
    ///
    /// The within-block analog of [`Self::assert_const`]: rather than holding at the whole block,
    /// each entry holds only at use-indices strictly greater than its stored index, recovering uses
    /// in the asserting block after the assert. The `Vec` is in instruction order (so the earliest
    /// establisher wins); it must not be sorted or deduplicated, as that order _is_ the
    /// deterministic first-writer rule.
    local_assert_const: HashMap<BlockId, Vec<(usize, ValueId, Arc<Constant>)>>,

    /// Equalities established by an `AssertCmp{Eq, a, b}` _within_ the keyed block (neither side a
    /// constant), each paired with the assert's instruction index.
    ///
    /// The within-block analog of [`Self::assert_eq`], holding only at use-indices strictly greater
    /// than the stored index; likewise kept in instruction order. Each pair is stored canonically
    /// as `(min, max)` by value id (the _entries_ stay in instruction order — only each pair's two
    /// sides are ordered).
    local_assert_eq: HashMap<BlockId, Vec<(usize, ValueId, ValueId)>>,

    /// Disequalities `(a, b)` from the false edge of an equality `JmpIf` dominating the block.
    diseq: HashMap<BlockId, HashSet<(ValueId, ValueId)>>,

    /// The honest values a free witness handle `r` is equal to.
    ///
    /// It is derived from the two readings of one correspondence: `r = witness_of(v)` — an unpinned
    /// `WriteWitness { result: Some(r), value: v, pinned: false }` (`r → v`) — and
    /// `v = value_of(r)` — a `Cast { value: r, target: ValueOf }` honest projection (`r → result`).
    /// Both key on the witness `r`; each `Vec` is sorted + deduplicated for determinism.
    witness_fwd: HashMap<ValueId, Vec<ValueId>>,

    /// Per-block, index-thresholded asserted-equal leaders, precomputed in [`build`] so
    /// [`Self::asserted_leader`] is a lookup rather than a per-query transitive-closure scan.
    ///
    /// For each value that joins a non-singleton asserted-equal class in the block, a list of
    /// `(threshold, leader)` steps sorted ascending by `threshold`. The leader in effect for a
    /// query at `point.index = q` is the last step with `threshold <= q`; no such step (or no entry
    /// for the value) means `v` is a singleton there, so the answer is `None`. A cross-block pair
    /// holds at every index (`threshold = 0`); a same-block pair at instruction index `i` holds
    /// only for use-indices `> i` (`threshold = i + 1`, so it applies iff `i < q`). Empty when the
    /// function has no asserted equalities.
    asserted_leaders: HashMap<BlockId, HashMap<ValueId, Vec<(usize, ValueId)>>>,

    /// Values pinned to a constant by a _bound-to-run_ assert.
    ///
    /// This is an assert in a block strictly post-dominating the keyed block (statically or over
    /// the executable subgraph, when the build's solver pruned an edge), whose values are
    /// single-valued per invocation or already final at the keyed block, and in scope there.
    ///
    /// The post-dominance analog of [`Self::assert_const`]. Anticipated facts are justified by the
    /// assert _still running_ on the genuine values (a run on which the fact is false aborts there,
    /// or hangs — both rejecting), so consumers must never use them to rewrite the operands of any
    /// `Assert`/`AssertCmp`.
    anticipated_const: HashMap<BlockId, HashMap<ValueId, Arc<Constant>>>,

    /// Equalities established by a bound-to-run `AssertCmp{Eq, a, b}` (with neither side a
    /// constant), under the same stability and scope gates.
    ///
    /// This is the post-dominance analog of [`Self::assert_eq`], with the same canonical `(min,
    /// max)` dedup and consumer contract as [`Self::anticipated_const`].
    anticipated_eq: HashMap<BlockId, Vec<(ValueId, ValueId)>>,

    /// Per-block leader tables over the _union_ of every equality holding in the keyed block: its
    /// dominance-direction pairs ([`Self::assert_eq`]), its own same-block pairs read in _both_
    /// index directions ([`Self::local_assert_eq`]), and its anticipated pairs
    /// ([`Self::anticipated_eq`]).
    ///
    /// Unlike [`Self::asserted_leaders`], the classes are index-independent: a same-block pair at
    /// instruction index `i` holds at every use-index — `> i` because its assert has already run,
    /// `< i` because it is bound to run on the very same bindings (straight-line, same-iteration,
    /// so neither stability nor scope needs checking) — and the one point neither direction covers,
    /// the establishing assert's own operands (`q == i`), is protected by Gate 3 alone (see the
    /// module "Index Granularity" section). The only remaining index-dependence is SSA scope of the
    /// _redirect target_: each member maps to a single `(threshold, leader)` — `leader` is its
    /// class's `DefKey` minimum, `threshold` is `0` when the leader is defined in a dominating
    /// block or bound at the keyed block's entry, else `d + 1` for a leader defined by the block's
    /// own instruction `d` (valid only at use-indices `> d`).
    ///
    /// One entry suffices (no step list): `DefKey` order lists dominating-block and entry-bound
    /// members before same-block instruction-defined ones (and the latter by instruction rank), so
    /// the class minimum is also the earliest-available member; and at any _real_ use of the
    /// queried value the minimum always passes its own threshold — its `DefKey` is at most the
    /// queried value's, whose definition precedes the use.
    ///
    /// Built only for blocks with at least one same-block or anticipated pair (a cross-only block's
    /// union table would equal its asserted table); [`Self::anticipated_leader`] falls back to the
    /// plain asserted table otherwise.
    anticipated_leaders: HashMap<BlockId, HashMap<ValueId, (usize, ValueId)>>,
}

impl ConditionalFacts {
    /// The shared shape of the two const queries.
    ///
    /// The `cross` (whole-block) map of the queried direction, then the `local_assert_const` scan
    /// at the indices `applies` admits. The `Vec` is in instruction order, so `find` returns the
    /// earliest establisher (first-writer-wins).
    fn const_at(
        &self,
        cross: &HashMap<BlockId, HashMap<ValueId, Arc<Constant>>>,
        point: ProgramPoint,
        v: ValueId,
        applies: impl Fn(usize) -> bool,
    ) -> Option<Arc<Constant>> {
        // A cross-block assert holds at every index of the block.
        if let Some(c) = cross.get(&point.block).and_then(|m| m.get(&v)) {
            return Some(c.clone());
        }
        self.local_assert_const
            .get(&point.block)?
            .iter()
            .find(|(i, vv, _)| applies(*i) && *vv == v)
            .map(|(_, _, c)| c.clone())
    }

    /// The shared shape of the two equality queries (reflexive + symmetric).
    ///
    /// The `cross` map of the queried direction, then the `local_assert_eq` scan at the indices
    /// `applies` admits. Pairs are stored canonically as `(min, max)` by value id, so the query
    /// pair is canonicalized once and compared directly.
    fn equal_at(
        &self,
        cross: &HashMap<BlockId, Vec<(ValueId, ValueId)>>,
        point: ProgramPoint,
        a: ValueId,
        b: ValueId,
        applies: impl Fn(usize) -> bool,
    ) -> bool {
        let (lo, hi) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        a == b
            || cross
                .get(&point.block)
                .is_some_and(|eqs| eqs.iter().any(|p| *p == (lo, hi)))
            || self.local_assert_eq.get(&point.block).is_some_and(|eqs| {
                eqs.iter()
                    .any(|(i, x, y)| applies(*i) && (*x, *y) == (lo, hi))
            })
    }

    /// The constant `v` is forced to at `point` by a dominating assert (valid at every index of the
    /// block) or by a same-block assert strictly before `point.index`, or `None`.
    pub(crate) fn asserted_const(&self, point: ProgramPoint, v: ValueId) -> Option<Arc<Constant>> {
        self.const_at(&self.assert_const, point, v, |i| i < point.index)
    }

    /// `true` if `a == b` is proven at `point` by a dominating assert or by a same-block assert
    /// strictly before `point.index` (reflexive + symmetric).
    pub(crate) fn asserted_equal(&self, point: ProgramPoint, a: ValueId, b: ValueId) -> bool {
        self.equal_at(&self.assert_eq, point, a, b, |i| i < point.index)
    }

    /// Gets the canonical representative of `v`'s transitive asserted-equal class at `point`.
    ///
    /// This is its _dominance-root-most_ member — or `None` if `v` is in no asserted equality there
    /// (nothing to copy-propagate to). May return `v` itself when `v` is already the root-most.
    ///
    /// The class is the transitive closure of the equality pairs holding at `point`: every
    /// cross-block pair (`assert_eq`, at every index of the block) plus the same-block pairs
    /// (`local_assert_eq`) established strictly before `point.index`. Transitive closure is what
    /// makes the leader well-defined — every member of a class returns the _same_ representative,
    /// the member minimizing the dominance-consistent `DefKey` (the dominance-earliest).
    ///
    /// Every component member's definition dominates `point`: each pair's operands are inputs of an
    /// `AssertCmp` that either dominates `point.block` or precedes `point` within it, so by SSA
    /// they are defined before `point`. The blocks dominating `point.block` form a single chain, so
    /// the members are mutually dominance-comparable and the `DefKey` minimum is the
    /// dominance-root-most — hence it itself dominates `point` and is available there (this is what
    /// makes the redirect to it sound).
    ///
    /// The classes are precomputed per block in [`build`] (see [`build_block_leaders`]); this is
    /// the lookup. `asserted_leaders[block][v]` holds the `(threshold, leader)` steps sorted
    /// ascending by `threshold`, and the leader in effect at `point.index` is the last step whose
    /// `threshold <= point.index`. Leaders are monotone in `threshold` (a class only grows, which
    /// can only lower the `DefKey` minimum), so the last applicable step is the current leader.
    pub(crate) fn asserted_leader(&self, point: ProgramPoint, v: ValueId) -> Option<ValueId> {
        let steps = self.asserted_leaders.get(&point.block)?.get(&v)?;
        let applicable = steps.partition_point(|(threshold, _)| *threshold <= point.index);
        (applicable > 0).then(|| steps[applicable - 1].1)
    }

    /// The constant `v` is pinned to at `point` by a _bound-to-run_ assert: one in a block
    /// strictly post-dominating `point.block` (stability and scope pre-gated in [`build`]), or one
    /// later in `point.block` — straight-line execution reaches a same-block assert on the same
    /// bindings, so the local direction needs no gates.
    ///
    /// This is the anticipated channel only; the dominance direction is [`Self::asserted_const`],
    /// and consumers chain the two. A consumer must never use an anticipated fact to rewrite the
    /// operands of any `Assert`/`AssertCmp` (see the `anticipated_const` field's docs).
    ///
    /// The local read is the mirror of `asserted_const`'s: an assert strictly _after_ the use
    /// anticipates the fact (`q == i` matches neither direction, so an assert never informs its
    /// own operands).
    pub(crate) fn anticipated_const(
        &self,
        point: ProgramPoint,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        self.const_at(&self.anticipated_const, point, v, |i| i > point.index)
    }

    /// `true` if `a == b` is anticipated at `point`: proven by a bound-to-run assert in a strictly
    /// post-dominating block, or by a same-block assert strictly after `point.index` (reflexive +
    /// symmetric). The anticipated mirror of [`Self::asserted_equal`], under the same consumer
    /// contract as [`Self::anticipated_const`].
    pub(crate) fn anticipated_equal(&self, point: ProgramPoint, a: ValueId, b: ValueId) -> bool {
        self.equal_at(&self.anticipated_eq, point, a, b, |i| i > point.index)
    }

    /// The canonical representative of `v`'s transitive equality class at `point` over the union of
    /// the block's dominance-direction, same-block (both index directions), and anticipated pairs —
    /// or `None` if `v` joins no class there.
    ///
    /// A block with same-block or anticipated pairs carries a union table that supersedes its plain
    /// asserted table (its classes are supersets of the asserted classes); a block without one
    /// falls back to [`Self::asserted_leader`]. The union classes are index-independent (see
    /// [`Self::anticipated_leaders`]); the only index check left is the leader's own redirect
    /// threshold, which any query at a real use of `v` passes. All class members' defining blocks
    /// lie on the block's dominator chain (local-pair operands feed an assert inside it, cross and
    /// anticipated pairs are dominance- respectively Gate-2-checked at insertion), so they are
    /// mutually dominance-comparable and the `DefKey` minimum is well-defined. A redirect produced
    /// by this query may be anticipated-justified, so it inherits the [`Self::anticipated_const`]
    /// consumer contract (never into `Assert`/`AssertCmp` operands) — which is also the sole guard
    /// at the establishing assert's own index.
    pub(crate) fn anticipated_leader(&self, point: ProgramPoint, v: ValueId) -> Option<ValueId> {
        let Some(table) = self.anticipated_leaders.get(&point.block) else {
            return self.asserted_leader(point, v);
        };
        let &(threshold, leader) = table.get(&v)?;
        (point.index >= threshold).then_some(leader)
    }

    /// `true` if `a` and `b` are proven unequal on entry to `bid` (symmetric).
    pub(crate) fn known_disequal(&self, bid: BlockId, a: ValueId, b: ValueId) -> bool {
        self.diseq
            .get(&bid)
            .is_some_and(|set| set.contains(&(a, b)) || set.contains(&(b, a)))
    }

    /// The honest values equal to the free witness handle `r` (the `WriteWitness` hint and every
    /// `value_of(r)` read), sorted + deduped; an empty slice if `r` forwards nothing.
    pub(crate) fn witness_forward(&self, r: ValueId) -> &[ValueId] {
        self.witness_fwd
            .get(&r)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

// BUILD
// ================================================================================================

/// Compute the conditional side facts of `function`.
///
/// ### Arguments
///
/// - `cyclic` is the function's static-CFG block cyclicity, computed once per function by the
///    caller and shared across the per-context rebuilds (it is context-independent).
/// - `reach` is the caller's static-CFG block reachability, precomputed only for functions that are
///    cyclic _and_ contain an assert, the sole shape whose finality gate consults it, and `None`
///    elsewhere.
/// - `det` is the Phase-0 per-`(callee, return)` determinism, feeding the stability gate's det-call
///   form (it is context-independent, like `cyclic`).
/// - `static_edge_count` is the function's deduped static terminator-edge count, computed once by
///   the caller (also context-independent): the baseline [`ExecView::build`] compares the solver's
///   executable-edge count against to detect the unpruned fast path.
///
/// `cyclic` and `reach` are the _unpruned fast path_: a build whose solver proved a static edge
/// non-executable derives its own sharper executable-subgraph view (see `exec_view.rs`) and reads
/// that instead.
pub(crate) fn build(
    function: &HLFunction,
    facts: &FunctionFacts,
    cfg: &CFG,
    consts: &HLSSAConstantsSnapshot,
    type_info: &FunctionTypeInfo,
    order: &DefOrder,
    cyclic: &CyclicBlocks,
    reach: Option<&BlockReach>,
    det: &DetSummaries,
    static_edge_count: usize,
) -> ConditionalFacts {
    let mut out = ConditionalFacts::default();

    // The _unconditional_ constant of `v` (interned or proven by the lattice), if any — including
    // aggregates, which callers must gate out before recording a fact.
    //
    // Asserts that pin a value to such a constant produce a _conditional_ fact; the unconditional
    // view is never touched.
    let const_of = |v: ValueId| -> Option<Arc<Constant>> {
        if let Some(c) = consts.get(&v) {
            return Some(c.clone());
        }
        match facts.lattice.get(&v) {
            Some(Constness::Const(c)) => Some(c.clone()),
            _ => None,
        }
    };

    // Step 1: Gather facts at their establishing block — tagged with the instruction index of the
    // establishing assert, so the asserting block's own post-assert uses can be recovered at index
    // granularity (the cross-block fan-out in Step 3 ignores the index) — plus the (global) witness
    // forwards and the `Cmp{Eq}` results needed to interpret equality branches.
    let mut raw_const: HashMap<BlockId, Vec<(usize, ValueId, Arc<Constant>)>> = HashMap::default();
    let mut raw_eq: HashMap<BlockId, Vec<(usize, ValueId, ValueId)>> = HashMap::default();
    let mut eq_cmp_of: HashMap<ValueId, (ValueId, ValueId)> = HashMap::default();

    for (bid, block) in function.get_blocks() {
        if !facts.reachable.contains(bid) {
            continue;
        }
        for (idx, instr) in block.get_instructions().enumerate() {
            match instr {
                OpCode::Assert { value } => {
                    raw_const
                        .entry(*bid)
                        .or_default()
                        .push((idx, *value, bool_constant(true)));
                }
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs,
                    rhs,
                } => {
                    // The pinned side, if the other is an unconditional constant: `(value, c)`.
                    let pin = const_of(*lhs)
                        .map(|c| (*rhs, c))
                        .or_else(|| const_of(*rhs).map(|c| (*lhs, c)));
                    match pin {
                        Some((v, c)) if c.is_scalar() => {
                            raw_const.entry(*bid).or_default().push((idx, v, c));
                        }

                        // An aggregate (`Blob`) pin is recorded in _neither_ channel: Blob
                        // constants are never surfaced to consumers (see `const_in_facts`), and
                        // routing the pair to the eq channel would put a constant-sided pair into
                        // the leader machinery.
                        Some(_) => {}
                        None => {
                            // Store the pair canonically (min, max) by value id so `(x, y)` and
                            // `(y, x)` dedup as one in the cross-block fan-out below.
                            let (lo, hi) = if lhs.0 <= rhs.0 {
                                (*lhs, *rhs)
                            } else {
                                (*rhs, *lhs)
                            };
                            raw_eq.entry(*bid).or_default().push((idx, lo, hi));
                        }
                    }
                }
                OpCode::Cmp {
                    kind: CmpKind::Eq,
                    result,
                    lhs,
                    rhs,
                } => {
                    eq_cmp_of.insert(*result, (*lhs, *rhs));
                }

                // A redirect `r → v` pins a free witness to its honest value (honest witgen sets
                // `r = v`): it _adds_ a functional constraint, so it narrows the accept set
                // (`A_M ⊆ A_N`) and never rejects the honest run.
                OpCode::WriteWitness {
                    result: Some(r),
                    value,
                    pinned: false,
                } if type_info.get_value_type(*r).is_witness_of() => {
                    out.witness_fwd.entry(*r).or_default().push(*value);
                }

                // The same correspondence read the other way: `result = value_of(r)` strips `r`'s
                // witness wrapper to its honest value, so `result == r` honestly — another member of
                // `r`'s honest-value set. The operand `value` is the witness (always `WitnessOf` for
                // a well-typed `ValueOf`; the guard mirrors the `WriteWitness` arm).
                OpCode::Cast {
                    result,
                    value,
                    target: CastTarget::ValueOf,
                } if type_info.get_value_type(*value).is_witness_of() => {
                    out.witness_fwd.entry(*value).or_default().push(*result);
                }
                _ => {}
            }
        }
    }

    // Canonicalize each witness's honest-value set: sort + dedup so the result is independent of
    // block/instruction scan order (the analysis must be byte-deterministic).
    for members in out.witness_fwd.values_mut() {
        members.sort_unstable();
        members.dedup();
    }

    // Step 2: Disequalities from the false edge of an equality branch, keyed by the false target at
    // whose entry they hold. The fact holds there only when that edge is the target's _sole_
    // executable in-edge (otherwise a join could merge in a path that never established it).
    let mut raw_diseq: HashMap<BlockId, HashSet<(ValueId, ValueId)>> = HashMap::default();

    // Executable in-edges per block (the reverse of `exec_edges`), built once for efficiency.
    let mut exec_preds_of: HashMap<BlockId, Vec<BlockId>> = HashMap::default();
    for (p, t) in &facts.exec_edges {
        exec_preds_of.entry(*t).or_default().push(*p);
    }

    for (bid, block) in function.get_blocks() {
        if !facts.reachable.contains(bid) {
            continue;
        }
        let Some(Terminator::JmpIf(cond, then_b, else_b)) = block.get_terminator() else {
            continue;
        };

        // A degenerate branch to the same block both ways establishes nothing on either edge.
        if then_b == else_b {
            continue;
        }
        let Some((lhs, rhs)) = eq_cmp_of.get(cond) else {
            continue;
        };
        if !facts.reachable.contains(else_b) {
            continue;
        }

        let preds = exec_preds_of
            .get(else_b)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if preds.len() == 1 && preds[0] == *bid {
            raw_diseq.entry(*else_b).or_default().insert((*lhs, *rhs));
        }
    }

    // Step 3: Fan out to dominated blocks. The inner loop only needs the blocks that actually
    // established a fact (the keys of the raw maps), not every reachable block. Iterating sorted
    // targets `C` (outer) and sorted fact-source blocks `B` (inner) keeps the result deterministic
    // and first-writer-wins on any contradictory pair of asserts, which is vacuous in both
    // directions: contradictory asserts can only co-dominate — or both post-dominate, and hence
    // both be bound to run from — a block that no accepting run reaches.
    let mut reachable_blocks: Vec<BlockId> = facts.reachable.iter().copied().collect();
    reachable_blocks.sort_unstable();
    let mut fact_sources: Vec<BlockId> = raw_const
        .keys()
        .chain(raw_eq.keys())
        .chain(raw_diseq.keys())
        .copied()
        .collect();
    fact_sources.sort_unstable();
    fact_sources.dedup();

    // Gate 1 (anticipated direction) — Value stability, in two parts (see `stability.rs`): the
    // binding the bound-to-run assert checks must _be_ the binding at the earlier point. Part A,
    // the per-value invariance closure (single-valued per invocation), is built per call because
    // its constant and congruence rules read this build's converged facts; Part B, per-point
    // binding finality — an unstable value whose defining block no non-empty path from `c`
    // re-enters already carries, at every index of `c`, the final binding the assert checks.
    //
    // Both gates are consulted only by the anticipated arm below, which reads `raw_const` /
    // `raw_eq` — so when the function establishes no assert fact at all, the closure (a full
    // function walk, per context) is skipped as dead weight.
    let have_assert_facts = !raw_const.is_empty() || !raw_eq.is_empty();

    // The executable-subgraph view, built per conditional build (see [`ExecView`] for why per
    // context matters) — but only when an assert fact exists to consume it AND this build's solver
    // actually pruned a static edge (`ExecView::build` returns `None` on the unpruned fast path,
    // where the caller's shared static structures are exact). This bounds the cost across numerous
    // contexts. The cyclicity and finality gates below simply read the sharper view when it exists:
    // executable cycles and paths are subsets of static ones, so each answer is a pointwise
    // superset.
    let exec = have_assert_facts
        .then(|| ExecView::build(function, facts, static_edge_count))
        .flatten();
    let cyclic: &CyclicBlocks = exec.as_ref().map_or(cyclic, |e| &e.cyclic);
    let reach: Option<&BlockReach> = exec.as_ref().map_or(reach, |e| e.reach.as_ref());

    debug_assert!(
        reach.is_some() || !cyclic.has_cycles() || !have_assert_facts,
        "ICE: The finality gate needs `reach` whenever assert facts exist in a cyclic view (static: guaranteed by the caller's gating; exec: built exactly when the view is cyclic)"
    );

    let mut stability = have_assert_facts.then(|| {
        ValueStability::compute(
            function,
            cfg,
            cyclic,
            order,
            &facts.congruence,
            |v| const_of(v).is_some(),
            |g, j| det_return(det, g, j),
        )
    });
    let mut stable_at = |v: ValueId, c: BlockId| -> bool {
        let stability = stability
            .as_mut()
            .expect("the anticipated arm fires only on fact sources, so the closure was built");
        stability.is_stable(v) || reach.is_some_and(|r| !r.reaches(c, order.def_block(v)))
    };

    // Gate 2 (anticipated direction) — SSA scope: post-dominance, unlike dominance, does not put
    // the fact's values in scope at the target, so each value's defining block must dominate it
    // (reflexively: a value defined in the target itself is queryable only at in-scope indices,
    // and `DefOrder`'s entry fallback admits entry parameters and interned constants everywhere).
    let in_scope_at = |v: ValueId, c: BlockId| -> bool { cfg.dominates(order.def_block(v), c) };

    for &c in &reachable_blocks {
        for &b in &fact_sources {
            // Assert facts hold at `C` when the assert has already run before `C`, i.e. `B`
            // strictly dominates `C`. `B != C` excludes the asserting block's own entry; its own
            // post-assert uses are recovered separately at index granularity from the `local_*`
            // maps (see the module "Index Granularity" section). Dominance makes the asserted value
            // transitively live at `C`, so no in-scope check is needed.
            if b != c && cfg.dominates(b, c) {
                if let Some(consts_at_b) = raw_const.get(&b) {
                    let entry = out.assert_const.entry(c).or_default();
                    for (_i, v, k) in consts_at_b {
                        entry.entry(*v).or_insert_with(|| k.clone());
                    }
                }
                if let Some(eqs_at_b) = raw_eq.get(&b) {
                    let entry = out.assert_eq.entry(c).or_default();
                    for (_i, x, y) in eqs_at_b {
                        let pair = (*x, *y);
                        if !entry.contains(&pair) {
                            entry.push(pair);
                        }
                    }
                }
            }

            // Disequalities are path-conditional edge facts already attached to their target's
            // entry, so they propagate reflexively (`B == C` keeps the fact at the target itself).
            if cfg.dominates(b, c) {
                if let Some(diseqs_at_b) = raw_diseq.get(&b) {
                    let entry = out.diseq.entry(c).or_default();
                    for pair in diseqs_at_b {
                        entry.insert(*pair);
                    }
                }
            }

            // The anticipated (post-dominance) direction: an assert in `B` is _bound to run_ at
            // every block `C` it strictly post-dominates — any run visiting `C` reaches the assert
            // (which checks the genuine values), aborts earlier, or hangs, and all non-reaching
            // outcomes reject. Gated per fact value on stability and scope (above). The relation is
            // the _union_ of the static answer and (when the solver pruned an edge) the
            // executable-subgraph answer — see the module doc and `exec_view.rs` for why each
            // disjunct is sound and neither subsumes the other. Disequalities stay out: a
            // branch-edge fact is path-conditional, not bound to run.
            if b != c
                && (cfg.post_dominates(b, c)
                    || exec.as_ref().is_some_and(|e| e.post_dominates(b, c)))
            {
                if let Some(consts_at_b) = raw_const.get(&b) {
                    for (_i, v, k) in consts_at_b {
                        if stable_at(*v, c) && in_scope_at(*v, c) {
                            out.anticipated_const
                                .entry(c)
                                .or_default()
                                .entry(*v)
                                .or_insert_with(|| k.clone());
                        }
                    }
                }
                if let Some(eqs_at_b) = raw_eq.get(&b) {
                    for (_i, x, y) in eqs_at_b {
                        if stable_at(*x, c)
                            && stable_at(*y, c)
                            && in_scope_at(*x, c)
                            && in_scope_at(*y, c)
                        {
                            let pair = (*x, *y);
                            let entry = out.anticipated_eq.entry(c).or_default();
                            if !entry.contains(&pair) {
                                entry.push(pair);
                            }
                        }
                    }
                }
            }
        }
    }

    // The raw per-block asserts (now unborrowed by Step 3) _are_ the asserting block's own facts,
    // already in instruction order: move them in to drive the index-granular `local_*` queries.
    out.local_assert_const = raw_const;
    out.local_assert_eq = raw_eq;

    // Step 4: Per-block, index-thresholded asserted-equal leaders, consumed by
    // `ConditionalFacts::asserted_leader`. Only values appearing in an equality pair can be class
    // members, and every such value appears in `local_assert_eq` (the cross-block `assert_eq` is
    // fanned out from the same pairs), so a non-empty `local_assert_eq` is the authoritative "any
    // equalities exist" gate — the per-block leader tables are built only then (the definition
    // `order` is shared from the caller, built once for the congruence leaders too).
    if out.local_assert_eq.values().any(|eqs| !eqs.is_empty()) {
        for &b in &reachable_blocks {
            let cross = out.assert_eq.get(&b).map(Vec::as_slice).unwrap_or(&[]);
            let local = out
                .local_assert_eq
                .get(&b)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let table = build_block_leaders(cross, local, &order);
            if !table.is_empty() {
                out.asserted_leaders.insert(b, table);
            }
        }
    }

    // Step 5: Per-block _anticipated_ leader tables — one transitive closure per block over the
    // union of its dominance pairs, its own same-block pairs, and its anticipated pairs.
    //
    // This accounts for both index directions. A same-block pair holds after its assert because it
    // has already run, and before it as long as it is bound to run on the very same bindings. The
    // classes need no per-pair thresholds (see the `anticipated_leaders` field docs) as the
    // per-leader redirect threshold enforces the SSA scope. Every member is in scope at the block —
    // anticipated pairs pass Gate 2's reflexive dominance check at insertion, and cross/local pairs
    // are in scope structurally (their operands feed an assert of a dominating block or of this
    // one) — so members defined by instructions inside the block itself are admitted too.
    //
    // Blocks with neither a same-block nor an anticipated pair get no table (a cross-only union
    // table would equal the asserted one), bounding the extra closure cost to blocks the new
    // direction actually reaches; `anticipated_leader` falls back to the plain asserted table
    // there.
    for &c in &reachable_blocks {
        let anticipated = out.anticipated_eq.get(&c).map(Vec::as_slice).unwrap_or(&[]);
        let local = out
            .local_assert_eq
            .get(&c)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        if anticipated.is_empty() && local.is_empty() {
            continue;
        }

        let mut pairs = out.assert_eq.get(&c).cloned().unwrap_or_default();
        pairs.extend(anticipated.iter().copied());
        pairs.extend(local.iter().map(|&(_i, a, b)| (a, b)));
        let table = build_union_block_leaders(&pairs, c, order);
        if !table.is_empty() {
            out.anticipated_leaders.insert(c, table);
        }
    }

    out
}

// INTERNAL FUNCTIONS
// ================================================================================================

/// Precompute the index-thresholded asserted-equal leaders of one block.
///
/// `cross` are the block's dominating equality pairs — they hold at every index, so they form the
/// threshold-`0` classes. `local` are its same-block pairs `(index, a, b)`, each holding only for
/// use-indices `> index`, so it is applied at threshold `index + 1` (matching the `i < point.index`
/// rule).
///
/// The result maps each value that ever joins a non-singleton class to its `(threshold, leader)`
/// steps in ascending `threshold` order, the leader being the `DefKey`-minimum of the value's
/// component once all pairs up to that threshold are merged. A value that stays a singleton —
/// including one appearing only in a degenerate `(a, a)` self-pair — gets no entry, so its leader
/// query is `None`, exactly as the previous per-query closure returned.
fn build_block_leaders(
    cross: &[(ValueId, ValueId)],
    local: &[(usize, ValueId, ValueId)],
    order: &DefOrder,
) -> HashMap<ValueId, Vec<(usize, ValueId)>> {
    // Local pairs arrive in strictly ascending index order — Step 1 pushes them during one forward
    // walk of the block's instructions, and indices are unique within a block — which is exactly
    // the ascending-threshold batch order the step lists need.
    debug_assert!(local.windows(2).all(|w| w[0].0 < w[1].0));

    // The last leader recorded for each value, so a batch that leaves a value's leader unchanged
    // adds no step.
    let mut last_leader: HashMap<ValueId, ValueId> = HashMap::default();
    let mut table: HashMap<ValueId, Vec<(usize, ValueId)>> = HashMap::default();

    // Cross pairs at threshold 0, then each local pair at its own `index + 1`, ascending.
    let mut uf: UnionFind<ValueId> = UnionFind::default();
    record_batch(0, cross, &mut uf, &mut last_leader, &mut table, order);
    for &(i, a, b) in local {
        record_batch(
            i + 1,
            &[(a, b)],
            &mut uf,
            &mut last_leader,
            &mut table,
            order,
        );
    }

    table
}

/// Precompute the index-independent union leader table of one block (see
/// [`ConditionalFacts::anticipated_leaders`]).
///
/// `pairs` is the union of the block's cross, anticipated, and (index-stripped) same-block equality
/// pairs; the classes are their one-shot transitive closure. Each non-singleton member maps to its
/// class's `DefKey`-minimum leader, thresholded by the leader's own definition site in `block` (`0`
/// unless the leader is defined by one of `block`'s instructions).
fn build_union_block_leaders(
    pairs: &[(ValueId, ValueId)],
    block: BlockId,
    order: &DefOrder,
) -> HashMap<ValueId, (usize, ValueId)> {
    let mut uf: UnionFind<ValueId> = UnionFind::default();
    for &(a, b) in pairs {
        uf.union(a, b);
    }
    let (reps, leader_of, size_of) = recover_partition(&mut uf, order);

    let mut table: HashMap<ValueId, (usize, ValueId)> = HashMap::default();
    for (v, leader) in non_singleton_leaders(&reps, &leader_of, &size_of) {
        table.insert(v, (order.redirect_threshold_in(leader, block), leader));
    }
    table
}

/// Recover the current partition of `uf` together with each class's `DefKey`-minimum leader and
/// size.
///
/// The partition is read back via `nodes()`/`find()` (as `points_to::array_cells` does): the
/// returned `reps` pair every tracked value with its class representative, and the two maps are
/// keyed by representative — the leader for redirect targets, the size for skipping singletons.
/// `DefKey` is a total order (value-id tie-break), so the minimum is unique and independent of
/// iteration order — the leader is deterministic.
#[allow(clippy::type_complexity)]
fn recover_partition(
    uf: &mut UnionFind<ValueId>,
    order: &DefOrder,
) -> (
    Vec<(ValueId, ValueId)>,
    HashMap<ValueId, ValueId>,
    HashMap<ValueId, usize>,
) {
    let nodes: Vec<ValueId> = uf.nodes().collect();
    let reps: Vec<(ValueId, ValueId)> = nodes.into_iter().map(|v| (v, uf.find(v))).collect();

    let mut leader_of: HashMap<ValueId, ValueId> = HashMap::default();
    let mut size_of: HashMap<ValueId, usize> = HashMap::default();
    for &(v, rep) in &reps {
        *size_of.entry(rep).or_insert(0) += 1;
        let cur = leader_of.entry(rep).or_insert(v);
        if order.key(v) < order.key(*cur) {
            *cur = v;
        }
    }

    (reps, leader_of, size_of)
}

/// Merge one batch of equality `pairs` into `uf` at `threshold`, then record a `(threshold,
/// leader)` step for every value whose class leader changed.
///
/// Classes still singleton (size `< 2`) are skipped, so a value never in a real two-distinct-value
/// class gets no step.
fn record_batch(
    threshold: usize,
    pairs: &[(ValueId, ValueId)],
    uf: &mut UnionFind<ValueId>,
    last_leader: &mut HashMap<ValueId, ValueId>,
    table: &mut HashMap<ValueId, Vec<(usize, ValueId)>>,
    order: &DefOrder,
) {
    for &(a, b) in pairs {
        uf.union(a, b);
    }
    let (reps, leader_of, size_of) = recover_partition(uf, order);

    // Record a step for each non-singleton member whose class leader changed at this threshold.
    for (v, leader) in non_singleton_leaders(&reps, &leader_of, &size_of) {
        if last_leader.get(&v) != Some(&leader) {
            table.entry(v).or_default().push((threshold, leader));
            last_leader.insert(v, leader);
        }
    }
}

/// The non-singleton members of a recovered partition, paired with their class's `DefKey`-minimum
/// leader.
///
/// Values in singleton classes — including those appearing only in a degenerate `(a, a)` self-pair
/// — are skipped: a value never in a real two-distinct-value class earns no leader entry, the
/// shared grounding rule of both leader tables.
fn non_singleton_leaders<'p>(
    reps: &'p [(ValueId, ValueId)],
    leader_of: &'p HashMap<ValueId, ValueId>,
    size_of: &'p HashMap<ValueId, usize>,
) -> impl Iterator<Item = (ValueId, ValueId)> + 'p {
    reps.iter()
        .filter(|(_, rep)| size_of[rep] >= 2)
        .map(|&(v, rep)| (v, leader_of[&rep]))
}
