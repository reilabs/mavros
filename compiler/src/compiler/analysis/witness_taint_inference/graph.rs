//! The `≥` constraint graph and its monotone-OR solver.
//!
//! An edge `a ≥ b` means "if `b` is Witness then `a` is Witness" — taint flows *from* `b` *to* `a`.
//! We store the reverse adjacency `taints: b ↦ { a : a ≥ b }` so that a forward BFS from a set of
//! forced-Witness seeds reaches exactly the Witness closure. Because the lattice is the 2-point
//! chain `Pure ≤ Witness` and every rule is monotone-OR, this reachability *is* the least fixed
//! point.
//!
//! The graph is generic over the node type for easier adaptability, with the added benefit of being
//! able to unit-test it in isolation.

use crate::collections::{HashMap, HashSet};

use std::{collections::VecDeque, hash::Hash};

/// A directed `≥` graph over nodes `N`.
#[derive(Clone, Debug)]
pub struct TaintGraph<N: Eq + Hash + Clone> {
    /// `taints[b]` = the set of `a` such that `a ≥ b` (i.e. the nodes `b` taints when it is a
    /// Witness).
    taints: HashMap<N, HashSet<N>>,
}

impl<N: Eq + Hash + Clone> TaintGraph<N> {
    pub fn new() -> Self {
        Self {
            taints: HashMap::default(),
        }
    }

    /// Add `a ≥ b`: "if `b` is Witness then `a` is Witness".
    ///
    /// Self-edges are dropped as a node is always `≥` itself, and they only slow the solver.
    pub fn add_ge(&mut self, a: N, b: N) {
        if a == b {
            return;
        }
        self.taints.entry(b).or_default().insert(a);
    }

    /// Add `a ≥ b` *and* `b ≥ a` — equality, used to unify aliased memory levels.
    pub fn add_eq(&mut self, a: N, b: N) {
        self.add_ge(a.clone(), b.clone());
        self.add_ge(b, a);
    }

    /// The least set of Witness nodes given that every node in `seeds` is forced to be Witness.
    ///
    /// This is the closure of `seeds` under "if `b` is Witness then every `a ≥ b` is Witness". The
    /// returned set always contains `seeds` themselves.
    pub fn solve<I: IntoIterator<Item = N>>(&self, seeds: I) -> HashSet<N> {
        let mut witness: HashSet<N> = HashSet::default();
        let mut queue: VecDeque<N> = VecDeque::new();
        for s in seeds {
            if witness.insert(s.clone()) {
                queue.push_back(s);
            }
        }
        while let Some(b) = queue.pop_front() {
            if let Some(deps) = self.taints.get(&b) {
                for a in deps {
                    if witness.insert(a.clone()) {
                        queue.push_back(a.clone());
                    }
                }
            }
        }
        witness
    }

    /// Multi-source reachability in a single traversal: for each node, which of `sources` taint it.
    ///
    /// Bit `i` of the returned bitset for node `n` is set iff `sources[i]` reaches `n` —
    /// equivalently iff `n ∈ self.solve([sources[i]])`. Each node carries a word-bitset of source
    /// indices that grows monotonically to a fixpoint (a node is re-queued whenever its bits grow),
    /// so all sources share one traversal instead of one BFS per source. Summary extraction uses it
    /// to read every `(sink, source)` pair from a single solve.
    pub fn reaching_sources(&self, sources: &[N]) -> HashMap<N, Vec<u64>> {
        let words = sources.len().div_ceil(64);
        let mut reaching: HashMap<N, Vec<u64>> = HashMap::default();
        let mut queue: VecDeque<N> = VecDeque::new();
        // Nodes currently pending in `queue`: a node whose bits grow several times before it is
        // popped needs only one pending visit, since the visit reads its latest bits.
        let mut in_queue: HashSet<N> = HashSet::default();

        for (i, s) in sources.iter().enumerate() {
            let bits = reaching
                .entry(s.clone())
                .or_insert_with(|| vec![0u64; words]);
            bits[i / 64] |= 1u64 << (i % 64);
            if in_queue.insert(s.clone()) {
                queue.push_back(s.clone());
            }
        }

        while let Some(b) = queue.pop_front() {
            in_queue.remove(&b);
            let Some(deps) = self.taints.get(&b) else {
                continue;
            };
            // Snapshot `b`'s bits so we can borrow `reaching` mutably for each dependent `a`.
            let b_bits = reaching
                .get(&b)
                .cloned()
                .unwrap_or_else(|| vec![0u64; words]);
            for a in deps {
                let bits = reaching
                    .entry(a.clone())
                    .or_insert_with(|| vec![0u64; words]);
                let mut changed = false;
                for w in 0..words {
                    let before = bits[w];
                    bits[w] |= b_bits[w];
                    changed |= bits[w] != before;
                }
                if changed && in_queue.insert(a.clone()) {
                    queue.push_back(a.clone());
                }
            }
        }
        reaching
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_yields_only_seeds() {
        let g: TaintGraph<u32> = TaintGraph::new();
        assert_eq!(g.solve([]).len(), 0);
        assert_eq!(g.solve([7]), HashSet::from_iter([7]));
    }

    #[test]
    fn ge_propagates_from_b_to_a() {
        // a ≥ b : seeding b taints a; seeding a does not taint b.
        let mut g: TaintGraph<u32> = TaintGraph::new();
        g.add_ge(1, 2); // 1 ≥ 2
        assert_eq!(g.solve([2]), HashSet::from_iter([1, 2]));
        assert_eq!(g.solve([1]), HashSet::from_iter([1]));
    }

    #[test]
    fn eq_propagates_both_ways() {
        let mut g: TaintGraph<u32> = TaintGraph::new();
        g.add_eq(1, 2);
        assert_eq!(g.solve([1]), HashSet::from_iter([1, 2]));
        assert_eq!(g.solve([2]), HashSet::from_iter([1, 2]));
    }

    #[test]
    fn chains_close_transitively() {
        // 3 ≥ 2 ≥ 1 : seeding 1 taints 2 and 3.
        let mut g: TaintGraph<u32> = TaintGraph::new();
        g.add_ge(2, 1);
        g.add_ge(3, 2);
        assert_eq!(g.solve([1]), HashSet::from_iter([1, 2, 3]));
        assert_eq!(g.solve([3]), HashSet::from_iter([3]));
    }

    #[test]
    fn cycles_terminate_at_least_fixed_point() {
        // f ≥ g and g ≥ f (mutual recursion shape): seeding either taints both, and it terminates.
        let mut g: TaintGraph<u32> = TaintGraph::new();
        g.add_ge(10, 11);
        g.add_ge(11, 10);
        assert_eq!(g.solve([10]), HashSet::from_iter([10, 11]));
        // A disconnected node stays Pure.
        g.add_ge(20, 21);
        assert!(!g.solve([10]).contains(&20));
    }

    #[test]
    fn reaching_sources_matches_per_source_solve() {
        // 3 ≥ 2 ≥ 1, plus a disconnected 5 ≥ 4.
        let mut g: TaintGraph<u32> = TaintGraph::new();
        g.add_ge(2, 1);
        g.add_ge(3, 2);
        g.add_ge(5, 4);

        let sources = vec![1u32, 4u32];
        let reaching = g.reaching_sources(&sources);
        let bit = |node: u32, i: usize| {
            reaching
                .get(&node)
                .map(|bits| bits[i / 64] & (1u64 << (i % 64)) != 0)
                .unwrap_or(false)
        };

        // Source 1 (index 0) taints the whole 1→2→3 chain; source 4 (index 1) only 4→5.
        assert!(bit(1, 0) && bit(2, 0) && bit(3, 0));
        assert!(!bit(2, 1) && !bit(3, 1));
        assert!(bit(4, 1) && bit(5, 1));
        assert!(!bit(4, 0) && !bit(5, 0));

        // The labeling agrees with `solve` run once per source.
        for (i, s) in sources.iter().enumerate() {
            let solved = g.solve([*s]);
            for node in [1u32, 2, 3, 4, 5] {
                assert_eq!(bit(node, i), solved.contains(&node));
            }
        }
    }

    #[test]
    fn top_source_forces_witness() {
        // Model an unconditional Witness source as a distinguished node seeded into solve().
        const TOP: u32 = 0;
        let mut g: TaintGraph<u32> = TaintGraph::new();
        g.add_ge(5, TOP); // 5 ≥ Top : 5 is always Witness
        assert!(g.solve([TOP]).contains(&5));
        assert!(!g.solve([]).contains(&5)); // without seeding Top, nothing forces it
    }
}
