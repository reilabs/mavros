//! The value-stability side of the anticipated-channel gates.
//!
//! An anticipated fact at block `C`, justified by an assert in a strictly post-dominating block
//! `B`, is sound **only** when each fact value's binding at `C` _is_ the binding the bound-to-run
//! assert checks.
//!
//! This module provides the two sufficient per-value conditions, which can be combined by the
//! consumer as `stable(v) || binding_final_at(v, C)`:
//!
//! # Part A: Invariance ([`ValueStability`])
//!
//! `v` is _single-valued per invocation_. Every binding event within one run binds the same value
//! (a strict generalization of "bound at most once"). The **grounded (least-fixpoint)** closure of
//! four rules:
//!
//! 1. `v` is an unconditional constant.
//! 2. `def_block(v)` lies outside every CFG cycle of the supplied view ([`CyclicBlocks`], static or
//!    executable): it executes at most once per invocation, so `v` is bound at most once. Entry
//!    formals share this form regardless of entry cyclicity: they are bound by the caller, never by
//!    a CFG edge — no argument-carrying edge targets the entry block (a `JmpIf` back edge carries
//!    no arguments, and nothing emits an argument-carrying `Jmp` at the entry; asserted in
//!    [`ValueStability::compute`]) — so re-entering the entry block never rebinds them.
//! 3. `v` results from a deterministic op whose operands are all stable. By the definition of an
//!    SSA, each operand is bound before the op executes and holds its unique per-invocation value,
//!    so every re-execution recomputes the same result. Determinism has two forms: a pure op
//!    ([`pure_op_operands`]: the _same_ single source of truth as value numbering and the
//!    determinism summaries, so witness and memory ops are out by construction), or a constrained
//!    static call whose return is a deterministic function of its arguments per the determinism
//!    summaries — the identical property the congruence `CallDet` numbering trusts. Dynamic and
//!    unconstrained calls have neither form (an advice result is pinned to no function of its
//!    arguments).
//! 4. `v` is congruent to a class member `w != v` with `def(w)` dominating `def(v)` and `w`
//!    stable **by rules 1–3 only**: at each binding event of `v`, `w` is bound (dominance) and
//!    equal to `v` at that instant — the same instant-equality that justifies congruence-leader
//!    redirects — and `w` is single-valued, so all of `v`'s bindings coincide.
//!
//! Rule 4 is deliberately restricted to rules-1–3-stable witnesses: a fixpoint that lets rule 4
//! feed itself would be coinductive and admit two parallel loop-carried φs of one header — which
//! genuinely share a congruence class — "each stable because the other is", though both are
//! multi-valued.
//!
//! Loop-carried block parameters are exactly what must stay unstable; rule 4's residual yield is a
//! deterministic call over _unstable_ operands congruent to a dominating duplicate (rule 4 needs no
//! operand stability, only instant-equality to a single-valued witness). Rules 1–3 converge in a
//! single dominator-preorder pass — SSA operand definitions dominate their uses, so every operand
//! bit is final before it is read, and both rule-3 forms read exactly those bits — and rule 4 is a
//! memoized post-check over the finished rules-1–3 outputs.
//!
//! # Part B: Binding Finality ([`BlockReach`])
//!
//! Even an unstable `v` is admissible at `C` when no non-empty CFG path leads from `C` back to
//! `def_block(v)`: Gate 2 (definition dominates `C`) puts the binding in scope at `C`'s entry,
//! and once control is in `C` the defining block never executes again, so the binding at every
//! index of `C` is the run's _final_ binding of `v` — the one every later program point, in
//! particular the bound-to-run assert, observes.
//!
//! The non-empty-path semantics make `reaches(c, c)` true exactly when `c` lies on a cycle, so a
//! same-block definition needs no special case: in an acyclic `c` rule 2 already grants stability,
//! and in a cyclic `c` the back-reach correctly rejects (an early-index query in iteration `k`
//! would see iteration `k−1`'s binding).
//!
//! Both parts are built from an explicit block/edge list ([`CyclicBlocks::from_edges`],
//! [`BlockReach::from_edges`]) and admit two views of one function. The **static** CFG view
//! ([`terminator_edges`]; shared per function) under-approximates admissibility: pruning
//! non-executable edges only removes cycles and paths — fewer facts, never a wrong one. The
//! **executable-subgraph** view (per conditional build, from the solver's `exec_edges`; see
//! [`super::exec_view`]) is strictly sharper and equally sound: an honest run traverses only
//! executable edges, so a revisit of a block traces a non-empty executable cycle through it (Part
//! A's rule 2) and a re-execution of `def_block(v)` after visiting `C` traces a non-empty
//! executable path `C → def_block(v)` (Part B).
//!
//! For an equality pair each member is gated independently, possibly by different parts: the
//! assert checks the genuine pair on bindings `X* = Y*`, and each member's gate independently
//! equates its binding at `C` with its checked binding, so the equality holds at `C`.

use petgraph::graph::{DiGraph, NodeIndex};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            click_cooper::{
                congruence::{Congruence, pure_op_operands},
                def_order::DefOrder,
            },
            flow_analysis::CFG,
        },
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{CallTarget, HLFunction, OpCode},
        },
    },
};

// VALUE STABILITY
// ================================================================================================

/// The per-value invariance closure.
///
/// Built once per conditional build. Rule 1 reads the per-context constant view and rule 4 the
/// per-context congruence, so the bits are not shareable across contexts.
pub(crate) struct ValueStability<'a> {
    /// An acyclic CFG makes every value stable by rule 2 alone, so the maps stay empty and every
    /// query answers `true` immediately.
    ///
    /// This is a fast path.
    all_stable: bool,

    /// The rules-1–3 stability bit of every value defined or referenced in a dominator-preorder
    /// block; a missing entry is conservatively unstable.
    ///
    /// Rule 4 draws its witnesses from **this** map only — never from the combined answer — so
    /// every derivation stays grounded and two parallel loop-carried φs can never justify each
    /// other (see the module doc).
    base_stable: HashMap<ValueId, bool>,

    /// The rule-4 (congruence-tier) memo, demand-filled by [`Self::is_stable`].
    ///
    /// Each answer is a pure function of `base_stable` and the congruence partition, so the fill
    /// order cannot influence any result.
    congruence_stable: HashMap<ValueId, bool>,

    /// The congruence results used by the stability definition.
    congruence: &'a Congruence,

    /// The definition order used by the stability definition.
    order: &'a DefOrder<'a>,
}

impl<'a> ValueStability<'a> {
    /// Compute the rules-1–3 bits of `function` in one dominator-preorder pass.
    ///
    /// ### Arguments
    ///
    /// - `cyclic` supplies rule 2's block-level primitive.
    /// - `is_const` reports rule 1 — whether a value is an unconditional constant (interned or
    ///   proven by this build's lattice).
    /// - `call_det` reports rule 3's det-call form — whether return `j` of a callee is a
    ///   deterministic function of its arguments (the determinism summaries).
    pub(crate) fn compute(
        function: &HLFunction,
        cfg: &CFG,
        cyclic: &CyclicBlocks,
        order: &'a DefOrder<'a>,
        congruence: &'a Congruence,
        is_const: impl Fn(ValueId) -> bool,
        call_det: impl Fn(FunctionId, usize) -> bool,
    ) -> Self {
        let mut this = Self {
            all_stable: !cyclic.has_cycles(),
            base_stable: HashMap::default(),
            congruence_stable: HashMap::default(),
            congruence,
            order,
        };
        if this.all_stable {
            return this;
        }

        // The rule-1/2 default of a value: a constant, or defined in an acyclic block (with
        // `DefOrder`'s entry fallback for unrecorded values — interned constants and the like).
        let default_bit =
            |v: ValueId| -> bool { is_const(v) || !cyclic.is_in_cycle(order.def_block(v)) };

        // The shared rule-1/2 prefix of every per-site form: constants and acyclic-block defs are
        // stable outright; `form` is the site's own admitting disjunct (rule 2's formal form for
        // parameters, rule 3's two forms for instruction results).
        let rule_bit = |v: ValueId, in_cycle: bool, form: bool| is_const(v) || !in_cycle || form;

        // Rule 2's formal form rests on a structural invariant, not merely on `JmpIf` carrying no
        // arguments: no argument-carrying edge ever targets the entry block, so nothing can rebind
        // its parameters. The frontend materialises every loop header as a fresh block, and no pass
        // points an argument-carrying `Jmp` at the entry.
        let entry = function.get_entry_id();
        #[cfg(debug_assertions)]
        for bid in cfg.get_domination_pre_order() {
            if let Some(Terminator::Jmp(target, args)) = function.get_block(bid).get_terminator() {
                debug_assert!(
                    *target != entry || args.is_empty(),
                    "an argument-carrying edge targets the entry block; rule 2's formal form is \
                     unsound for rebindable entry parameters"
                );
            }
        }

        // Dominator preorder guarantees every recorded operand definition is visited before its
        // uses (SSA definitions dominate uses; parameters rank before a block's instructions), so
        // rules 1–3 need exactly one pass. Referenced-but-unrecorded values are seeded with the
        // rule-1/2 default so rule-4 witness lookups can resolve them later.
        for bid in cfg.get_domination_pre_order() {
            let block = function.get_block(bid);
            let in_cycle = cyclic.is_in_cycle(bid);

            for p in block.get_parameter_values() {
                // Rule 2, formal form: entry parameters are bound once by the caller and never
                // rebound by a CFG edge (asserted above), so entry cyclicity is irrelevant to them.
                let bit = rule_bit(*p, in_cycle, bid == entry);
                this.base_stable.insert(*p, bit);
            }
            for instr in block.get_instructions() {
                for v in instr.get_inputs() {
                    if !this.base_stable.contains_key(v) {
                        this.base_stable.insert(*v, default_bit(*v));
                    }
                }

                // Rule 3, det-call form: a constrained static call whose return `j` is a
                // deterministic function of its arguments rebinds `r_j` to the same value on every
                // re-execution — the pure-op argument, with the determinism supplied
                // interprocedurally by the summaries instead of by `pure_op_operands`. Dynamic and
                // unconstrained calls fall through to the pure-op form, which classifies no call,
                // so they keep their rule-1/2-only bits.
                if let OpCode::Call {
                    results,
                    function: CallTarget::Static(g),
                    args,
                    unconstrained: false,
                } = instr
                {
                    let args_stable = args
                        .iter()
                        .all(|a| this.base_stable.get(a).copied().unwrap_or(false));
                    for (j, r) in results.iter().enumerate() {
                        let bit = rule_bit(*r, in_cycle, args_stable && call_det(*g, j));
                        this.base_stable.insert(*r, bit);
                    }
                    continue;
                }

                // Rule 3, pure-op form: a deterministic op over stable operands rebinds the same
                // value on every re-execution, so all of its results are single-valued.
                let invariant = pure_op_operands(instr).is_some_and(|ops| {
                    ops.iter()
                        .all(|o| this.base_stable.get(o).copied().unwrap_or(false))
                });
                for r in instr.get_results() {
                    let bit = rule_bit(*r, in_cycle, invariant);
                    this.base_stable.insert(*r, bit);
                }
            }
            let term_args: &[ValueId] = match block.get_terminator() {
                Some(Terminator::Jmp(_, args)) | Some(Terminator::Return(args)) => args,
                Some(Terminator::JmpIf(cond, _, _)) => std::slice::from_ref(cond),
                None => &[],
            };
            for v in term_args {
                if !this.base_stable.contains_key(v) {
                    this.base_stable.insert(*v, default_bit(*v));
                }
            }
        }
        this
    }

    /// `true` if `v` is single-valued per invocation.
    pub(crate) fn is_stable(&mut self, v: ValueId) -> bool {
        if self.all_stable || self.base_stable.get(&v).copied().unwrap_or(false) {
            return true;
        }
        if let Some(&bit) = self.congruence_stable.get(&v) {
            return bit;
        }
        // Rule 4: a dominating, rules-1–3-stable congruent witness. `class_members` is sorted,
        // and `any` over a pure predicate is iteration-order-free.
        let bit = self.congruence.class_members(v).iter().any(|&w| {
            w != v
                && self.order.dominates_def(w, v)
                && self.base_stable.get(&w).copied().unwrap_or(false)
        });
        self.congruence_stable.insert(v, bit);
        bit
    }
}

// TERMINATOR EDGES
// ================================================================================================

/// Every `from → to` edge of `function`'s static CFG, read off the block terminators in block
/// order: `Jmp` contributes one edge, `JmpIf` two (possibly coincident), `Return` and unterminated
/// blocks none.
///
/// The static edge list behind the [`CyclicBlocks::new`] / [`BlockReach::new`] views, and the
/// pruning baseline of the executable-subgraph view (the solver's `exec_edges` is always a subset
/// of this list; see [`super::exec_view`]).
pub(crate) fn terminator_edges(function: &HLFunction) -> Vec<(BlockId, BlockId)> {
    let mut edges = Vec::new();
    for (bid, block) in function.get_blocks() {
        match block.get_terminator() {
            Some(Terminator::Jmp(t, _)) => edges.push((*bid, *t)),
            Some(Terminator::JmpIf(_, then_b, else_b)) => {
                edges.push((*bid, *then_b));
                edges.push((*bid, *else_b));
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }
    edges
}

/// The deduped static terminator-edge count of `function`: the baseline of the executable view's
/// unpruned fast path (see [`super::exec_view`]). A `HashSet` dedups a degenerate `JmpIf(c, t, t)`
/// exactly as the solver's executable-edge set does, so the counts compare like-for-like.
pub(crate) fn static_edge_count(function: &HLFunction) -> usize {
    terminator_edges(function)
        .into_iter()
        .collect::<HashSet<_>>()
        .len()
}

// BLOCK GRAPH
// ================================================================================================

/// Build a petgraph view of an explicit block/edge list: one node per block in `blocks` iteration
/// order (a sorted caller therefore gets deterministic node indices), one edge per pair — every
/// edge carrying `weight` — plus the two node↔block mappings.
///
/// Shared by [`CyclicBlocks::from_edges`] and the executable-subgraph post-dominance construction
/// (see [`super::exec_view`]). Every edge endpoint must be a supplied block.
pub(crate) fn block_graph<E: Clone>(
    blocks: impl IntoIterator<Item = BlockId>,
    edges: &[(BlockId, BlockId)],
    weight: E,
) -> (
    DiGraph<(), E>,
    HashMap<BlockId, NodeIndex<u32>>,
    Vec<BlockId>,
) {
    let mut graph: DiGraph<(), E> = DiGraph::new();
    let mut node_of: HashMap<BlockId, NodeIndex<u32>> = HashMap::default();
    let mut block_of: Vec<BlockId> = Vec::new();
    for bid in blocks {
        node_of.insert(bid, graph.add_node(()));
        block_of.push(bid);
    }
    for (from, to) in edges {
        graph.add_edge(node_of[from], node_of[to], weight.clone());
    }
    (graph, node_of, block_of)
}

// BLOCK REACHABILITY
// ================================================================================================

/// Non-empty-path reachability between the blocks of one CFG view of a function.
///
/// The static view is built **once per (cyclic, assert-bearing) function** and shared across every
/// per-context conditional build; the executable-subgraph view is per-build (see
/// [`super::exec_view`]). The result is consumed by membership only — no ordering is ever derived
/// from it — so internal iteration order cannot leak into compilation determinism (the same
/// argument as [`CyclicBlocks`]).
///
/// The per-source sweep is O(V·(V+E)) time and O(V²) space; should this ever profile hot, the
/// upgrade path is SCC condensation with `FixedBitSet` rows filled in reverse topological order,
/// O((V+E)·V/64).
pub(crate) struct BlockReach {
    /// `from → { every block a CFG path of length ≥ 1 leads to }`.
    reachable_from: HashMap<BlockId, HashSet<BlockId>>,
}

impl BlockReach {
    /// Compute the non-empty-path reachability of every block of `function`'s static CFG.
    pub(crate) fn new(function: &HLFunction) -> Self {
        Self::from_edges(
            function.get_blocks().map(|(bid, _)| *bid),
            &terminator_edges(function),
        )
    }

    /// Compute the non-empty-path reachability of an explicit CFG view.
    ///
    /// `blocks` must cover every node of the view — edge-free ones included, so each legitimate
    /// source gets a row and [`Self::reaches`]'s conservative unknown-source arm fires only for
    /// genuinely foreign blocks. Duplicate edges (a degenerate `JmpIf(c, t, t)`) are harmless: the
    /// flood dedups via its `seen` set.
    pub(crate) fn from_edges(
        blocks: impl IntoIterator<Item = BlockId>,
        edges: &[(BlockId, BlockId)],
    ) -> Self {
        let mut succs: HashMap<BlockId, Vec<BlockId>> = HashMap::default();
        for bid in blocks {
            succs.insert(bid, Vec::new());
        }
        for (from, to) in edges {
            succs
                .get_mut(from)
                .expect("every edge source is a supplied block")
                .push(*to);
        }

        // One flood per source, seeded from its _successors_ so the path is non-empty:
        // `reaches(b, b)` holds exactly when `b` lies on a cycle. Rows are independent and each is
        // a pure function of the edge set, so the source iteration order is immaterial.
        let mut reachable_from: HashMap<BlockId, HashSet<BlockId>> = HashMap::default();
        for (&bid, seeds) in &succs {
            let mut seen: HashSet<BlockId> = HashSet::default();
            let mut stack: Vec<BlockId> = seeds.clone();
            while let Some(b) = stack.pop() {
                if seen.insert(b) {
                    if let Some(next) = succs.get(&b) {
                        stack.extend(next.iter().copied());
                    }
                }
            }
            reachable_from.insert(bid, seen);
        }
        Self { reachable_from }
    }

    /// `true` if a CFG path of length ≥ 1 leads `from → … → to` (so `reaches(b, b)` iff `b` lies
    /// on a cycle). An unknown `from` conservatively answers `true` (rejecting the fact); it
    /// cannot occur for the reachable fan-out targets this is queried with.
    pub(crate) fn reaches(&self, from: BlockId, to: BlockId) -> bool {
        self.reachable_from
            .get(&from)
            .map_or(true, |set| set.contains(&to))
    }
}

// CYCLIC BLOCKS
// ================================================================================================

/// The blocks of one CFG view of a function that some cycle passes through: members of a
/// nontrivial SCC, or blocks that jump straight back to themselves.
///
/// A block outside every cycle executes at most once per function invocation, so its definitions
/// are bound at most once per run — the block-level primitive behind invariance rule 2. The static
/// view is computed once per function by ClickCooper (its only consumer) rather than in every CFG
/// build; the executable-subgraph view is per-build (see [`super::exec_view`]). The result is
/// consumed by membership only — no ordering is ever derived from the SCC iteration, so petgraph's
/// traversal order cannot leak into compilation determinism.
pub(crate) struct CyclicBlocks {
    blocks: HashSet<BlockId>,
}

impl CyclicBlocks {
    /// Classify every block of `function`'s static CFG.
    pub(crate) fn new(function: &HLFunction) -> Self {
        Self::from_edges(
            function.get_blocks().map(|(bid, _)| *bid),
            &terminator_edges(function),
        )
    }

    /// Classify every block of an explicit CFG view.
    ///
    /// `blocks` must cover every node of the view — edge-free ones included. Duplicate edges (a
    /// degenerate `JmpIf(c, t, t)`) are harmless: parallel edges cannot change the SCCs.
    pub(crate) fn from_edges(
        blocks: impl IntoIterator<Item = BlockId>,
        edges: &[(BlockId, BlockId)],
    ) -> Self {
        let (graph, _, block_of) = block_graph(blocks, edges, ());

        let mut blocks = HashSet::default();
        for scc in petgraph::algo::tarjan_scc(&graph) {
            if scc.len() > 1 || graph.find_edge(scc[0], scc[0]).is_some() {
                blocks.extend(scc.iter().map(|n| block_of[n.index()]));
            }
        }
        Self { blocks }
    }

    /// `true` if some CFG cycle passes through `block_id`: it belongs to a nontrivial strongly
    /// connected component, or it jumps back to itself.
    pub(crate) fn is_in_cycle(&self, block_id: BlockId) -> bool {
        self.blocks.contains(&block_id)
    }

    /// `true` if _any_ CFG cycle exists in this function — i.e. some block [`Self::is_in_cycle`].
    ///
    /// An acyclic CFG executes every block at most once per invocation, so every value is bound
    /// at most once per run.
    pub(crate) fn has_cycles(&self) -> bool {
        !self.blocks.is_empty()
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::HLSSA;

    #[test]
    fn two_block_cycle_reaches_itself() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();
        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_id();
        let a = f.add_block();
        let b = f.add_block();
        let exit = f.add_block();
        f.get_entry_mut().set_terminator(Terminator::Jmp(a, vec![]));
        f.get_block_mut(a)
            .set_terminator(Terminator::JmpIf(cond, b, exit));
        f.get_block_mut(b)
            .set_terminator(Terminator::Jmp(a, vec![]));
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![]));

        let reach = BlockReach::new(ssa.get_unique_entrypoint());
        // Both cycle members re-reach themselves; the entry and exit do not.
        assert!(reach.reaches(a, a));
        assert!(reach.reaches(b, b));
        assert!(reach.reaches(b, a));
        assert!(!reach.reaches(entry, entry));
        assert!(!reach.reaches(exit, exit));
        assert!(reach.reaches(entry, exit));
        assert!(!reach.reaches(exit, a));
        assert!(!reach.reaches(a, entry));

        // Cyclicity agrees: exactly the loop's SCC `{a, b}` is cyclic.
        let cyclic = CyclicBlocks::new(ssa.get_unique_entrypoint());
        assert!(cyclic.has_cycles());
        assert!(cyclic.is_in_cycle(a));
        assert!(cyclic.is_in_cycle(b));
        assert!(!cyclic.is_in_cycle(entry));
        assert!(!cyclic.is_in_cycle(exit));
    }

    #[test]
    fn diamond_has_no_back_reach() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();
        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_id();
        let left = f.add_block();
        let right = f.add_block();
        let join = f.add_block();
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, left, right));
        f.get_block_mut(left)
            .set_terminator(Terminator::Jmp(join, vec![]));
        f.get_block_mut(right)
            .set_terminator(Terminator::Jmp(join, vec![]));
        f.get_block_mut(join)
            .set_terminator(Terminator::Return(vec![]));

        let reach = BlockReach::new(ssa.get_unique_entrypoint());
        assert!(reach.reaches(entry, join));
        assert!(reach.reaches(left, join));
        assert!(!reach.reaches(left, right));
        assert!(!reach.reaches(left, left));
        assert!(!reach.reaches(join, entry));

        // An acyclic diamond has no cyclic block at all.
        let cyclic = CyclicBlocks::new(ssa.get_unique_entrypoint());
        assert!(!cyclic.has_cycles());
        for b in [entry, left, right, join] {
            assert!(!cyclic.is_in_cycle(b), "block {b:?} is acyclic");
        }
    }

    #[test]
    fn self_loop_reaches_itself() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();
        let f = ssa.get_unique_entrypoint_mut();
        let looping = f.add_block();
        let tail = f.add_block();
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(looping, vec![]));
        f.get_block_mut(looping)
            .set_terminator(Terminator::JmpIf(cond, looping, tail));
        f.get_block_mut(tail)
            .set_terminator(Terminator::Return(vec![]));

        let reach = BlockReach::new(ssa.get_unique_entrypoint());
        assert!(reach.reaches(looping, looping));
        assert!(!reach.reaches(tail, tail));
        assert!(!reach.reaches(tail, looping));

        // A single-node SCC with a self-edge still loops.
        let cyclic = CyclicBlocks::new(ssa.get_unique_entrypoint());
        assert!(cyclic.is_in_cycle(looping));
        assert!(!cyclic.is_in_cycle(tail));
    }

    /// A pruned edge list drops exactly the cycles and paths the removed edges carried — the
    /// executable-subgraph use of `from_edges` (see `exec_view`).
    #[test]
    fn pruned_edge_list_drops_cycles_and_paths() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();
        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_id();
        let a = f.add_block();
        let b = f.add_block();
        let exit = f.add_block();
        f.get_entry_mut().set_terminator(Terminator::Jmp(a, vec![]));
        f.get_block_mut(a)
            .set_terminator(Terminator::JmpIf(cond, b, exit));
        f.get_block_mut(b)
            .set_terminator(Terminator::Jmp(a, vec![]));
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![]));

        let f = ssa.get_unique_entrypoint();
        let blocks: Vec<BlockId> = f.get_blocks().map(|(bid, _)| *bid).collect();
        let all_edges = terminator_edges(f);

        // The full explicit view agrees with the terminator-derived one (delegation guard).
        let full_cyclic = CyclicBlocks::from_edges(blocks.iter().copied(), &all_edges);
        assert!(full_cyclic.is_in_cycle(a) && full_cyclic.is_in_cycle(b));

        // Prune the back edge `b → a` (as if the loop condition were constant-false): the cycle
        // vanishes, no block re-reaches itself, and the exit stays reachable.
        let pruned: Vec<_> = all_edges.iter().copied().filter(|&e| e != (b, a)).collect();
        let cyclic = CyclicBlocks::from_edges(blocks.iter().copied(), &pruned);
        assert!(!cyclic.has_cycles());
        let reach = BlockReach::from_edges(blocks.iter().copied(), &pruned);
        assert!(!reach.reaches(b, a));
        assert!(!reach.reaches(a, a));
        assert!(reach.reaches(entry, exit));
    }

    /// A self-looping block with no path to the exit must still be flagged cyclic — its
    /// definitions rebind per iteration regardless of the block being absent from the reverse
    /// dominator tree.
    #[test]
    fn no_exit_path_block_is_cyclic() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();
        let f = ssa.get_unique_entrypoint_mut();
        let dead = f.add_block();
        let out = f.add_block();
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, dead, out));
        f.get_block_mut(dead)
            .set_terminator(Terminator::Jmp(dead, vec![]));
        f.get_block_mut(out)
            .set_terminator(Terminator::Return(vec![]));

        let cyclic = CyclicBlocks::new(ssa.get_unique_entrypoint());
        assert!(cyclic.is_in_cycle(dead));
        assert!(!cyclic.is_in_cycle(out));
        assert!(!cyclic.is_in_cycle(ssa.get_unique_entrypoint().get_entry_id()));
    }
}
