//! Facts that are downstream of conditional expressions (asserts, equalities, disequalities, and
//! unpinned witness forwarding).
//!
//! These facts only hold on _accepting_ runs because a constraint establishes them. Per the
//! module-level soundness contract they are sound only under _constraint-preserving_ use (local
//! rewrites that maintain the establishing constraint). The computation here is performed
//! post-convergence using a single pass over the already solved function facts structure, plus the
//! dominance.
//!
//! # Dominance and Post-Dominance, not Dataflow
//!
//! An assert is a *global* constraint, not a runtime check: on an accepting run it holds at every
//! program point, whether the assert lies in that point's past or future. So an assert in reachable
//! block `B` contributes its fact to the entry of every reachable block `C` for which the assert is
//! guaranteed to be on the accepting run that flows through `C` — and there are two static ways to
//! prove that:
//!
//! - **`B` strictly dominates `C`** — the assert has *already run* before `C`. Static dominance is
//!   sound because every statically-dominating block also dominates on the executable subgraph
//!   (pruning paths only adds dominators), a conservative under-approximation of executable-path
//!   dominance, and thus subsumes the intersect-at-join dataflow used by branch facts.
//! - **`B` strictly post-dominates `C`** — the assert is *bound to run* on every continuation out of
//!   `C`, so on an accepting (terminating) run the constraint is satisfied and the fact holds at `C`
//!   too. (A non-terminating run never reaches the assert, but never reaches `exit` either, so it is
//!   non-accepting and excluded; `post_dominates` is `false` for a block that cannot reach `exit`,
//!   which is the safe answer.)
//!
//! The post-dominance direction needs one guard the dominance direction supplies for free: under
//! dominance `def(v) dom B dom C`, so the asserted value `v` is transitively live at `C`; under
//! post-dominance that chain breaks, so `v` (both sides, for an equality) must be independently
//! checked to be in scope at `C` (defined in a block that dominates `C`). This also keeps a
//! loop-carried value — defined inside a loop and asserted below it — from being mis-attributed to a
//! pre-loop block.
//!
//! Only asserts get the post-dominance direction. A disequality is an *edge* fact (the false edge of
//! an equality branch): it is path-conditional — true only on the taken edge — so it holds at the
//! entry of its target block `else_b` only when that edge is the block's sole executable in-edge, and
//! from there propagates to every block `else_b` *dominates* (reflexively). A post-dominated block
//! need not have taken that edge, so disequalities stay dominance-only.

use std::sync::Arc;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            click_cooper::{
                lattice::{Constness, bool_constant},
                solver::FunctionFacts,
            },
            flow_analysis::CFG,
            types::FunctionTypeInfo,
            value_definitions::{FunctionValueDefinitions, ValueDefinition},
        },
        ssa::{
            BlockId, Terminator, ValueId,
            hlssa::{CastTarget, CmpKind, Constant, HLFunction, HLSSAConstantsSnapshot, OpCode},
        },
    },
};

// CONDITIONAL FACTS
// ================================================================================================

/// The conditional facts of one function.
///
/// All maps are keyed by the block at whose **entry** the fact holds.
#[derive(Debug, Default)]
pub(crate) struct ConditionalFacts {
    /// Values forced to a constant by a dominating or post-dominating assert: `Assert{v}` ⇒
    /// `v = true`, `AssertCmp{Eq, x, c}` (with one side unconditional-constant `c`) ⇒ the other
    /// side `= c`.
    assert_const: HashMap<BlockId, HashMap<ValueId, Arc<Constant>>>,

    /// Equalities established by a dominating or post-dominating `AssertCmp{Eq, a, b}` where neither
    /// side is a (unconditional) constant — a *conditional* analog of congruence's `known_equal`.
    assert_eq: HashMap<BlockId, Vec<(ValueId, ValueId)>>,

    /// Disequalities `(a, b)` from the false edge of an equality `JmpIf` dominating the block.
    diseq: HashMap<BlockId, HashSet<(ValueId, ValueId)>>,

    /// The honest values a free witness handle `r` is equal to.
    ///
    /// It is derived from the two readings of one correspondence: `r = witness_of(v)` — an unpinned
    /// `WriteWitness { result: Some(r), value: v, pinned: false }` (`r → v`) — and
    /// `v = value_of(r)` — a `Cast { value: r, target: ValueOf }` honest projection (`r → result`).
    /// Both key on the witness `r`; each `Vec` is sorted + deduplicated for determinism.
    witness_fwd: HashMap<ValueId, Vec<ValueId>>,
}

impl ConditionalFacts {
    /// The constant `v` is forced to on entry to `bid` by a dominating assert, or `None`.
    pub(crate) fn asserted_const(&self, bid: BlockId, v: ValueId) -> Option<Arc<Constant>> {
        self.assert_const.get(&bid)?.get(&v).cloned()
    }

    /// `true` if a dominating assert proves `a == b` on entry to `bid` (reflexive + symmetric).
    pub(crate) fn asserted_equal(&self, bid: BlockId, a: ValueId, b: ValueId) -> bool {
        a == b
            || self.assert_eq.get(&bid).is_some_and(|eqs| {
                eqs.iter()
                    .any(|(x, y)| (*x == a && *y == b) || (*x == b && *y == a))
            })
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
) -> ConditionalFacts {
    let mut out = ConditionalFacts::default();

    // The *unconditional* constant of `v` (interned or proven by the lattice), if any.
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

    // A value→defining-block lookup, used to gate the post-dominance fan-out (Step 3) on the
    // asserted value being live at the target block. `v` is in scope on entry to `c` iff it is
    // defined in a block that dominates `c`. A value with no structural definition (an interned
    // constant / global) is in scope everywhere. Dominance fan-out needs no such check: `def(v)`
    // dominates the asserting block, which dominates the target, so `v` is transitively live there.
    let defs = FunctionValueDefinitions::from_function(function);
    let in_scope = |v: ValueId, c: BlockId| match defs.get_definition(v) {
        Some(ValueDefinition::Param(b, ..) | ValueDefinition::Instruction(b, ..)) => {
            cfg.dominates(*b, c)
        }
        None => true,
    };

    // Step 1: Gather facts at their establishing block, plus the (global) witness forwards and the
    // `Cmp{Eq}` results needed to interpret equality branches.
    let mut raw_const: HashMap<BlockId, Vec<(ValueId, Arc<Constant>)>> = HashMap::default();
    let mut raw_eq: HashMap<BlockId, Vec<(ValueId, ValueId)>> = HashMap::default();
    let mut eq_cmp_of: HashMap<ValueId, (ValueId, ValueId)> = HashMap::default();

    for (bid, block) in function.get_blocks() {
        if !facts.reachable.contains(bid) {
            continue;
        }
        for instr in block.get_instructions() {
            match instr {
                OpCode::Assert { value } => {
                    raw_const
                        .entry(*bid)
                        .or_default()
                        .push((*value, bool_constant(true)));
                }
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs,
                    rhs,
                } => {
                    if let Some(c) = const_of(*lhs) {
                        raw_const.entry(*bid).or_default().push((*rhs, c));
                    } else if let Some(c) = const_of(*rhs) {
                        raw_const.entry(*bid).or_default().push((*lhs, c));
                    } else {
                        raw_eq.entry(*bid).or_default().push((*lhs, *rhs));
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

        let preds: Vec<BlockId> = facts
            .exec_edges
            .iter()
            .filter(|(_, t)| t == else_b)
            .map(|(p, _)| *p)
            .collect();
        if preds.len() == 1 && preds[0] == *bid {
            raw_diseq.entry(*else_b).or_default().insert((*lhs, *rhs));
        }
    }

    // Step 3: Fan out to dominated (and, for asserts, post-dominated) blocks. The inner loop only
    // needs the blocks that actually established a fact (the keys of the raw maps), not every
    // reachable block. Iterating sorted targets `C` (outer) and sorted fact-source blocks `B`
    // (inner) keeps the result deterministic and first-writer-wins on any contradictory pair of
    // asserts (which can only co-dominate/co-post-dominate a block that no honest run reaches).
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
            // Assert facts hold at `C` when the assert is guaranteed on every run flowing through
            // `C`: either it has already run (`B` strictly dominates `C`) or it is bound to run on
            // every continuation (`B` strictly post-dominates `C`). `B != C` excludes the asserting
            // block's own entry.
            let dominates = b != c && cfg.dominates(b, c);
            let post_dominates = b != c && cfg.post_dominates(b, c);

            if dominates || post_dominates {
                if let Some(consts_at_b) = raw_const.get(&b) {
                    let entry = out.assert_const.entry(c).or_default();
                    for (v, k) in consts_at_b {
                        // Dominance makes `v` transitively live at `C`; under post-dominance only,
                        // `v` must be independently in scope (see the module soundness note).
                        if dominates || in_scope(*v, c) {
                            entry.entry(*v).or_insert_with(|| k.clone());
                        }
                    }
                }
                if let Some(eqs_at_b) = raw_eq.get(&b) {
                    let entry = out.assert_eq.entry(c).or_default();
                    for pair in eqs_at_b {
                        let (x, y) = pair;

                        // Post-dominance requires *both* sides of the equality to be in scope at
                        // `C`.
                        if (dominates || (in_scope(*x, c) && in_scope(*y, c)))
                            && !entry.contains(pair)
                        {
                            entry.push(*pair);
                        }
                    }
                }
            }

            // Disequalities are path-conditional edge facts already attached to their target's
            // entry, so they propagate reflexively (`B == C` keeps the fact at the target itself)
            // and stay dominance-only — a post-dominated block need not have taken that edge.
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

    out
}
