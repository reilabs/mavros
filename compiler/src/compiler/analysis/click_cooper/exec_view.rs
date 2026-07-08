//! The executable-subgraph view behind the anticipated (post-dominance) assert channel.
//!
//! One conditional build's converged [`FunctionFacts`] carries the solver's `reachable` blocks and
//! `exec_edges` — with the SCCP invariant that an edge absent from `exec_edges` provably never
//! executes (its branch condition folded the other way, or its source is unreachable). Every honest
//! run therefore traverses executable edges only, which makes three strictly sharper relations
//! sound for the anticipated channel:
//!
//! - **Post-Dominance:** If `B` post-dominates `C` in the executable subgraph, every executable
//!   path `C → exit` passes `B`, so a run visiting `C` reaches `B`'s assert, aborts earlier, or
//!   hangs — the same trichotomy that justifies the static relation. Note this is not a pointwise
//!   superset of static post-dominance: pruning can remove a block's only path to the exit,
//!   dropping it from the reverse traversal entirely (conservative `false` here). The consumer
//!   therefore takes the **union** `static || exec`, sound because each disjunct is.
//! - **Cyclicity (Invariance Rule 2):** A block outside every executable cycle cannot be visited
//!   twice by an honest run — a revisit traces a non-empty executable cycle through it.
//! - **Finality (Part B):** No non-empty executable path `C → def_block(v)` means the defining
//!   block never re-executes after control reaches `C`, so every index of `C` observes the run's
//!   final binding of `v`.
//!
//! Unreachable-def hygiene: stability bits computed for blocks outside `reachable` are never
//! load-bearing. Every value that can pass the consumer's SSA-scope gate at a reachable target has
//! a statically-dominating def block, and any static dominator of an exec-reachable block is itself
//! exec-reachable (the solver's executable entry-to-block chain is a static path, and dominators
//! lie on every static path); rule-3 operands and rule-4 witnesses dominate their dependents, so
//! the same argument covers them too.
//!
//! Determinism: node numbering is insertion-ordered (blocks sorted by id, then the virtual exit)
//! and edges are added from a sorted list, so [`CFGData::from_graph`]'s sorted-children traversal
//! sees identical indices across runs. The cyclicity and reachability results are consumed by
//! membership only (see their docs in [`super::stability`]).

use petgraph::graph::NodeIndex;

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            click_cooper::{
                solver::FunctionFacts,
                stability::{BlockReach, CyclicBlocks, block_graph},
            },
            flow_analysis::{CFGData, JumpType},
        },
        ssa::{BlockId, Terminator, hlssa::HLFunction},
    },
};

// EXEC VIEW
// ================================================================================================

/// The executable-subgraph refinement of the anticipated channel's CFG-shaped gates, built from
/// one conditional build's converged [`FunctionFacts`] — per function intraprocedurally, per
/// `(function, context)` contextually.
///
/// The per-context builds are where the static structures can never catch up: a 1-CFA-pinned
/// parameter that folds a branch cannot be rewritten into the shared function body, so only this
/// view sees the pruning.
pub(crate) struct ExecView {
    /// Reverse-graph dominators over the reachable blocks, the executable edges, and a virtual
    /// exit fed by every reachable `Return`-terminated block. Post-dominance == dominance here.
    postdom: CFGData,

    /// Block → node index in `postdom`'s graph (indices assigned over sorted block ids).
    node_of: HashMap<BlockId, NodeIndex<u32>>,

    /// Executable-edge cyclicity: invariance rule 2 and the `all_stable` fast path.
    pub(crate) cyclic: CyclicBlocks,

    /// Executable-edge non-empty-path reachability (the finality gate); built only when the view
    /// is cyclic, mirroring the caller's static gating (an acyclic view answers every stability
    /// query `true` before the finality gate is consulted).
    pub(crate) reach: Option<BlockReach>,
}

impl ExecView {
    /// Build the executable-subgraph view of one conditional build, or `None` on the unpruned
    /// fast path: when every static terminator edge is executable, the executable subgraph _is_
    /// the static CFG and the caller's shared static structures are exact.
    ///
    /// `static_edge_count` is the function's deduped static terminator-edge count, precomputed
    /// once per function by the caller (it is context-independent, and rebuilding the static edge
    /// set here would repeat per `(function, context)` build).
    pub(crate) fn build(
        function: &HLFunction,
        facts: &FunctionFacts,
        static_edge_count: usize,
    ) -> Option<Self> {
        // The solver discovers edges by walking reachable blocks' terminators, so `exec_edges` is
        // always a subset of the static terminator edges — equal _set_ sizes mean nothing was
        // pruned (a `HashSet` dedups a degenerate `JmpIf(c, t, t)` identically on both sides, so
        // the caller's deduped count compares like-for-like).
        #[cfg(debug_assertions)]
        {
            use crate::{
                collections::HashSet,
                compiler::analysis::click_cooper::stability::{self, terminator_edges},
            };
            debug_assert_eq!(
                stability::static_edge_count(function),
                static_edge_count,
                "the caller's static-edge count must match this function's terminators"
            );
            let static_edges: HashSet<(BlockId, BlockId)> =
                terminator_edges(function).into_iter().collect();
            debug_assert!(
                facts.exec_edges.is_subset(&static_edges),
                "solver executable edges must come from block terminators"
            );
        }

        if facts.exec_edges.len() == static_edge_count {
            return None;
        }

        // Deterministic node numbering: the reachable blocks sorted by id, then the virtual exit
        // (`CFGData` sorts dominator children by node index, so insertion order is load-bearing),
        // with edges added from a sorted list. Both endpoints of every executable edge are
        // reachable (the solver marks a target reachable the moment its in-edge is discovered),
        // so `block_graph`'s lookups cannot miss.
        let mut blocks: Vec<BlockId> = facts.reachable.iter().copied().collect();
        blocks.sort_unstable();
        let mut edges: Vec<(BlockId, BlockId)> = facts.exec_edges.iter().copied().collect();
        edges.sort_unstable();
        let (mut graph, node_of, _) = block_graph(blocks.iter().copied(), &edges, JumpType::Jmp);

        // Only reachable `Return` blocks feed the virtual exit. A `None`-terminated or
        // all-out-edges-pruned block becomes a dead end, conservatively absent from the reverse
        // traversal — the executable analog of a block with no static path to the exit.
        let exit = graph.add_node(());
        for &bid in &blocks {
            let term = function.get_block(bid).get_terminator();
            if matches!(term, Some(Terminator::Return(_))) {
                graph.add_edge(node_of[&bid], exit, JumpType::Return);
            }
        }

        // Post-dominance is dominance in the reversed graph rooted at the virtual exit — the same
        // construction as `flow_analysis`'s `CFGBuilder::build`.
        graph.reverse();
        let postdom = CFGData::from_graph(graph, exit);

        let cyclic = CyclicBlocks::from_edges(blocks.iter().copied(), &edges);
        let reach = cyclic
            .has_cycles()
            .then(|| BlockReach::from_edges(blocks.iter().copied(), &edges));

        Some(Self {
            postdom,
            node_of,
            cyclic,
            reach,
        })
    }

    /// `true` if `b` post-dominates `c` in the executable subgraph.
    ///
    /// Conservatively `false` when either block is missing from the reverse traversal —
    /// unreachable, or with no executable path to the exit (the executable analog of
    /// `flow_analysis`'s conservative answer for exit-less blocks). The consumer's union with the
    /// static relation recovers whatever the static view still grants there.
    pub(crate) fn post_dominates(&self, b: BlockId, c: BlockId) -> bool {
        let (Some(&nb), Some(&nc)) = (self.node_of.get(&b), self.node_of.get(&c)) else {
            return false;
        };
        self.postdom.dominates(nb, nc)
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::click_cooper::stability::{static_edge_count, terminator_edges},
        ssa::hlssa::HLSSA,
    };

    /// entry —JmpIf→ {then_b, else_b} —Jmp→ merge —Return; returns the ids.
    fn diamond() -> (HLSSA, BlockId, BlockId, BlockId, BlockId) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();
        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_id();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        f.get_block_mut(merge)
            .set_terminator(Terminator::Return(vec![]));
        (ssa, entry, then_b, else_b, merge)
    }

    #[test]
    fn unpruned_view_is_none() {
        let (ssa, entry, then_b, else_b, merge) = diamond();
        let function = ssa.get_unique_entrypoint();
        let mut facts = FunctionFacts::default();
        facts.reachable = [entry, then_b, else_b, merge].into_iter().collect();
        facts.exec_edges = terminator_edges(function).into_iter().collect();
        assert!(ExecView::build(function, &facts, static_edge_count(function)).is_none());
    }

    /// With the else edge pruned, the taken arm post-dominates the entry — the fact static
    /// post-dominance can never grant — while the pruned arm answers conservatively.
    #[test]
    fn pruned_diamond_grants_arm_post_dominance() {
        let (ssa, entry, then_b, else_b, merge) = diamond();
        let function = ssa.get_unique_entrypoint();
        let mut facts = FunctionFacts::default();
        facts.reachable = [entry, then_b, merge].into_iter().collect();
        facts.exec_edges = [(entry, then_b), (then_b, merge)].into_iter().collect();

        let view = ExecView::build(function, &facts, static_edge_count(function))
            .expect("an edge was pruned");
        assert!(view.post_dominates(then_b, entry));
        assert!(view.post_dominates(merge, entry));
        assert!(view.post_dominates(merge, then_b));
        assert!(!view.post_dominates(entry, merge));
        // The pruned arm is out of the view entirely: conservative `false` both ways.
        assert!(!view.post_dominates(else_b, entry));
        assert!(!view.post_dominates(merge, else_b));
        // An acyclic view carries no finality table (`all_stable` answers first).
        assert!(!view.cyclic.has_cycles());
        assert!(view.reach.is_none());
    }

    /// A reachable region with no executable path to the exit is absent from the reverse
    /// traversal: post-dominance answers conservatively `false`, while cyclicity and finality
    /// still see the live cycle.
    #[test]
    fn no_exec_exit_path_is_conservative() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();
        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_id();
        let dead = f.add_block();
        let live = f.add_block();
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, dead, live));
        f.get_block_mut(dead)
            .set_terminator(Terminator::Jmp(dead, vec![]));
        f.get_block_mut(live)
            .set_terminator(Terminator::Return(vec![]));

        let function = ssa.get_unique_entrypoint();
        let mut facts = FunctionFacts::default();
        facts.reachable = [entry, dead].into_iter().collect();
        facts.exec_edges = [(entry, dead), (dead, dead)].into_iter().collect();

        let view = ExecView::build(function, &facts, static_edge_count(function))
            .expect("the live edge was pruned");
        assert!(!view.post_dominates(dead, entry));
        assert!(!view.post_dominates(live, entry));
        assert!(view.cyclic.is_in_cycle(dead));
        assert!(!view.cyclic.is_in_cycle(entry));
        let reach = view.reach.as_ref().expect("the view is cyclic");
        assert!(reach.reaches(entry, dead));
        assert!(reach.reaches(dead, dead));
        assert!(!reach.reaches(entry, entry));
    }
}
