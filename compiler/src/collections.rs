//! Customized collections for the compiler.
//!
//! # HashMap and HashSet
//!
//! `std`'s default [`RandomState`](std::collections::hash_map::RandomState) seeds every map
//! differently per process, making iteration order nondeterministic. The compilation pipeline
//! allocates identifiers from monotonic counters (e.g. `SSA::fresh_value`) *while iterating* such
//! maps, so random iteration order leaks into emitted identifiers and ultimately into every dump
//! and artifact. Since the pipeline is single-threaded, fixing the hasher makes compilation a pure
//! function of its input.
//!
//! All compiler code must use these aliases instead of `std::collections::{HashMap, HashSet}`; this
//! is enforced by `disallowed-types` in the workspace `clippy.toml`.
//!
//! Notes:
//!
//! - Construct with `HashMap::default()`, not `::new()` — the inherent `new` only exists for
//!   the `RandomState` hasher.
//! - FxHash output depends on the target word size, so iteration order may differ between 32- and
//!   64-bit hosts. Run-to-run determinism on one host is the goal here; cross-architecture
//!   reproducibility is out of scope.

use std::hash::Hash;

pub use rustc_hash::FxBuildHasher;

// HASHED COLLECTIONS
// ================================================================================================

#[allow(clippy::disallowed_types)]
pub type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

#[allow(clippy::disallowed_types)]
pub type HashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

// DISJOINT-SET (UNION-FIND)
// ================================================================================================

/// A small, deterministic union-find (disjoint-set) over arbitrary copyable keys.
///
/// Keys are added lazily on first touch ([`Self::add`]/[`Self::find`]/[`Self::union`] all insert as
/// needed), so a caller that discovers its element set while walking some structure need not know
/// the element count up front — unlike index-based union-finds such as petgraph's `UnionFind`,
/// which require a dense `0..n` and `new(n)`.
#[derive(Debug, Clone)]
pub struct UnionFind<T> {
    parent: HashMap<T, T>,
}

impl<T> Default for UnionFind<T> {
    fn default() -> Self {
        // Written by hand rather than derived: the derive would demand `T: Default`, but an empty
        // union-find only needs `HashMap<T, T>: Default`, which holds for every `T`.
        Self {
            parent: HashMap::default(),
        }
    }
}

impl<T> UnionFind<T>
where
    T: Copy + Eq + Hash + Ord,
{
    /// Add `v` as its own singleton set if it is not already present.
    pub fn add(&mut self, v: T) {
        self.parent.entry(v).or_insert(v);
    }

    /// Return the representative of `v`'s set, compressing the path to it (path-halving).
    pub fn find(&mut self, v: T) -> T {
        self.add(v);
        let mut root = v;
        loop {
            let parent = self.parent[&root];
            if parent == root {
                break;
            }
            let grand = self.parent[&parent];
            // Path halving: point `root` at its grandparent, then continue from there.
            *self.parent.get_mut(&root).unwrap() = grand;
            root = grand;
        }
        root
    }

    /// Merge the sets containing `a` and `b`. The `Ord`-smaller representative becomes the new root,
    /// making the resulting partition deterministic.
    pub fn union(&mut self, a: T, b: T) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            let (root, child) = if ra <= rb { (ra, rb) } else { (rb, ra) };
            *self.parent.get_mut(&child).unwrap() = root;
        }
    }

    /// Iterate over every key currently tracked (in no particular order).
    pub fn nodes(&self) -> impl Iterator<Item = T> + '_ {
        self.parent.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::UnionFind;

    #[test]
    fn union_find_merges_and_finds_deterministically() {
        let mut uf = UnionFind::default();
        uf.union(3u32, 1);
        uf.union(1, 2);
        // All three share a root, and it is the smallest id (deterministic).
        assert_eq!(uf.find(3), 1);
        assert_eq!(uf.find(2), 1);
        assert_eq!(uf.find(1), 1);
        // A disconnected node is its own root.
        assert_eq!(uf.find(9), 9);
    }
}
