//! Facts that are downstream of conditional expressions (asserts, equalities, disequalities, and
//! unpinned witness forwarding).
//!
//! These facts only hold on _accepting_ runs because a constraint establishes them. Per the
//! module-level soundness contract they are sound only under _constraint-preserving_ use (local
//! rewrites that maintain the establishing constraint). The computation here is performed
//! post-convergence using a single pass over the already solved function facts structure, plus the
//! dominance.
//!
//! # Dominance, not Dataflow
//!
//! An assert is a *global* constraint, not a runtime check: on an accepting run it holds at every
//! program point. So an assert in reachable block `B` contributes its fact to the entry of every
//! reachable block `C` that `B` strictly dominates — the assert has *already run* before `C`, so
//! the constraint is established there. Static dominance is sound because every statically-dominating
//! block also dominates on the executable subgraph (pruning paths only adds dominators), a
//! conservative under-approximation of executable-path dominance, and thus subsumes the
//! intersect-at-join dataflow used by branch facts. Because `def(v) dom B dom C`, the asserted value
//! `v` is transitively live at `C`, so the dominance direction needs no separate in-scope check.
//!
//! # Index Granularity Within the Asserting Block
//!
//! The dominance fan-out above is *strict* (`B != C`), so it never attributes an assert to its own
//! block `B`. A use in `B` *after* the assert is instead recovered at index granularity: each
//! assert is recorded against `B` together with its instruction index (the `local_*` maps), and a
//! query at a program point in `B` sees the fact only when its use-index is strictly greater than
//! the assert's.
//!
//! Strict `>` is the index-granular form of the "already ran" (dominance) direction, and it
//! self-protects the constraint: an assert never informs its *own* operands (same index), so it
//! cannot be folded into a vacuous tautology — not even by a later, contradictory assert in `B`
//! (the earliest establisher wins — first-writer-wins). The asserted operands are the assert's own
//! inputs, hence defined before it, hence in scope at every later index — so the `local_*` path
//! needs no `in_scope` check.
//!
//! A use textually *before* a same-block assert is deliberately left at entry granularity: it still
//! sees any cross-block fact for the same value, and recovering the before-the-assert case would
//! reintroduce the contradiction hazard for no real gain.
//!
//! A disequality is an *edge* fact (the false edge of an equality branch): it is path-conditional —
//! true only on the taken edge — so it holds at the entry of its target block `else_b` only when
//! that edge is the block's sole executable in-edge, and from there propagates to every block
//! `else_b` *dominates* (reflexively).

use std::sync::Arc;

use crate::{
    collections::{HashMap, HashSet, UnionFind},
    compiler::{
        analysis::{
            click_cooper::{
                def_order::DefOrder,
                lattice::{Constness, bool_constant},
                solver::FunctionFacts,
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
/// The cross-block maps ([`Self::assert_const`], [`Self::assert_eq`], [`Self::diseq`]) are keyed by
/// the block at whose **entry** the fact holds. The two `local_*` maps are keyed by the *asserting*
/// block and additionally carry the establishing assert's instruction index, so their facts hold
/// only at use-indices after it (see their field docs and the module "Index Granularity" section).
#[derive(Debug, Default)]
pub(crate) struct ConditionalFacts {
    /// Values forced to a constant by a dominating assert: `Assert{v}` ⇒ `v = true`,
    /// `AssertCmp{Eq, x, c}` (with one side an unconditional *scalar* constant `c`) ⇒ the other
    /// side `= c`.
    ///
    /// Aggregate (`Blob`) pins are recorded nowhere as Blob constants are never surfaced to
    /// consumers.
    assert_const: HashMap<BlockId, HashMap<ValueId, Arc<Constant>>>,

    /// Equalities established by a dominating `AssertCmp{Eq, a, b}` where neither side is a
    /// (unconditional) constant — a *conditional* analog of congruence's `known_equal`.
    ///
    /// Each pair is stored canonically as `(min, max)` by value id, so `(a, b)` and `(b, a)` dedup
    /// as one.
    assert_eq: HashMap<BlockId, Vec<(ValueId, ValueId)>>,

    /// Values pinned to a constant by an assert *within* the keyed block, each paired with the
    /// instruction index of the establishing assert.
    ///
    /// The within-block analog of [`Self::assert_const`]: rather than holding at the whole block,
    /// each entry holds only at use-indices strictly greater than its stored index, recovering uses
    /// in the asserting block after the assert. The `Vec` is in instruction order (so the earliest
    /// establisher wins); it must not be sorted or deduplicated, as that order *is* the
    /// deterministic first-writer rule.
    local_assert_const: HashMap<BlockId, Vec<(usize, ValueId, Arc<Constant>)>>,

    /// Equalities established by an `AssertCmp{Eq, a, b}` *within* the keyed block (neither side a
    /// constant), each paired with the assert's instruction index.
    ///
    /// The within-block analog of [`Self::assert_eq`], holding only at use-indices strictly greater
    /// than the stored index; likewise kept in instruction order. Each pair is stored canonically
    /// as `(min, max)` by value id (the *entries* stay in instruction order — only each pair's two
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
}

impl ConditionalFacts {
    /// The constant `v` is forced to at `point` by a dominating assert (valid at every index of the
    /// block) or by a same-block assert strictly before `point.index`, or `None`.
    pub(crate) fn asserted_const(&self, point: ProgramPoint, v: ValueId) -> Option<Arc<Constant>> {
        // A cross-block assert holds at every index of the block.
        if let Some(c) = self.assert_const.get(&point.block).and_then(|m| m.get(&v)) {
            return Some(c.clone());
        }

        // Otherwise a same-block assert strictly *before* the use supplies the fact. The `Vec` is
        // in instruction order, so `find` returns the earliest establisher (first-writer-wins).
        self.local_assert_const
            .get(&point.block)?
            .iter()
            .find(|(i, vv, _)| *i < point.index && *vv == v)
            .map(|(_, _, c)| c.clone())
    }

    /// `true` if `a == b` is proven at `point` by a dominating assert or by a same-block assert
    /// strictly before `point.index` (reflexive + symmetric).
    pub(crate) fn asserted_equal(&self, point: ProgramPoint, a: ValueId, b: ValueId) -> bool {
        // Pairs are stored canonically as `(min, max)` by value id, so canonicalize the query pair
        // once and compare directly.
        let (lo, hi) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        a == b
            || self
                .assert_eq
                .get(&point.block)
                .is_some_and(|eqs| eqs.iter().any(|p| *p == (lo, hi)))
            || self.local_assert_eq.get(&point.block).is_some_and(|eqs| {
                eqs.iter()
                    .any(|(i, x, y)| *i < point.index && (*x, *y) == (lo, hi))
            })
    }

    /// Gets the canonical representative of `v`'s transitive asserted-equal class at `point`.
    ///
    /// This is its *dominance-root-most* member — or `None` if `v` is in no asserted equality there
    /// (nothing to copy-propagate to). May return `v` itself when `v` is already the root-most.
    ///
    /// The class is the transitive closure of the equality pairs holding at `point`: every
    /// cross-block pair (`assert_eq`, at every index of the block) plus the same-block pairs
    /// (`local_assert_eq`) established strictly before `point.index`. Transitive closure is what
    /// makes the leader well-defined — every member of a class returns the *same* representative,
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
pub(crate) fn build(
    function: &HLFunction,
    facts: &FunctionFacts,
    cfg: &CFG,
    consts: &HLSSAConstantsSnapshot,
    type_info: &FunctionTypeInfo,
    order: &DefOrder,
) -> ConditionalFacts {
    let mut out = ConditionalFacts::default();

    // The *unconditional* constant of `v` (interned or proven by the lattice), if any — including
    // aggregates, which callers must gate out before recording a fact.
    //
    // Asserts that pin a value to such a constant produce a *conditional* fact; the unconditional
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

                        // An aggregate (`Blob`) pin is recorded in *neither* channel: Blob
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
                // `r = v`): it *adds* a functional constraint, so it narrows the accept set
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
    // whose entry they hold. The fact holds there only when that edge is the target's *sole*
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
    // and first-writer-wins on any contradictory pair of asserts (which can only co-dominate a
    // block that no honest run reaches).
    let mut reachable_blocks: Vec<BlockId> = facts.reachable.iter().copied().collect();
    reachable_blocks.sort_by_key(|b| b.0);
    let mut fact_sources: Vec<BlockId> = raw_const
        .keys()
        .chain(raw_eq.keys())
        .chain(raw_diseq.keys())
        .copied()
        .collect();
    fact_sources.sort_unstable();
    fact_sources.dedup();

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
        }
    }

    // The raw per-block asserts (now unborrowed by Step 3) *are* the asserting block's own facts,
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
    // Local indices are unique within a block (one instruction per index), so sorting by index
    // gives a total, deterministic batch order.
    let mut sorted_local: Vec<(usize, ValueId, ValueId)> = local.to_vec();
    sorted_local.sort_by_key(|(i, _, _)| *i);

    // The last leader recorded for each value, so a batch that leaves a value's leader unchanged
    // adds no step.
    let mut last_leader: HashMap<ValueId, ValueId> = HashMap::default();
    let mut table: HashMap<ValueId, Vec<(usize, ValueId)>> = HashMap::default();

    // Cross pairs at threshold 0, then each local pair at its own `index + 1`, ascending.
    let mut uf: UnionFind<ValueId> = UnionFind::default();
    record_batch(0, cross, &mut uf, &mut last_leader, &mut table, order);
    for &(i, a, b) in &sorted_local {
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

    // Recover the current partition via `nodes()`/`find()` (as `points_to::array_cells` does):
    // every tracked value paired with its class representative.
    let nodes: Vec<ValueId> = uf.nodes().collect();
    let reps: Vec<(ValueId, ValueId)> = nodes.into_iter().map(|v| (v, uf.find(v))).collect();

    // Per class: its `DefKey`-minimum leader and its size (to skip singletons). `DefKey` is a total
    // order (value-id tie-break), so the minimum is unique and independent of iteration order — the
    // recorded leader is deterministic.
    let mut leader_of: HashMap<ValueId, ValueId> = HashMap::default();
    let mut size_of: HashMap<ValueId, usize> = HashMap::default();
    for &(v, rep) in &reps {
        *size_of.entry(rep).or_insert(0) += 1;
        let cur = leader_of.entry(rep).or_insert(v);
        if order.key(v) < order.key(*cur) {
            *cur = v;
        }
    }

    // Record a step for each non-singleton member whose class leader changed at this threshold.
    for &(v, rep) in &reps {
        if size_of[&rep] < 2 {
            continue;
        }
        let leader = leader_of[&rep];
        if last_leader.get(&v) != Some(&leader) {
            table.entry(v).or_default().push((threshold, leader));
            last_leader.insert(v, leader);
        }
    }
}
