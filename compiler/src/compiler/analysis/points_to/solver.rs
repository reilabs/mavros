//! The inclusion-based points-to constraint set and its worklist solver.
//!
//! Andersen-style points-to is dynamic transitive closure: `load`/`store` constraints add fresh
//! copy edges *during* solving as the pointer's points-to set grows, so it cannot pre-materialize
//! its edge set. It uses a worklist with an `in_queue` dedup set, and set values that only ever
//! grow to a least fixpoint.
//!
//! Each node's value is a [`HashSet`] of [`AbstractObject`]s, whose `==` is content equality
//! (order-independent), which the Phase-1 summary `!=` convergence relies on. The order is *not*
//! canonical; no result here depends on it, but a downstream consumer that emits IR per pointee
//! should sort first.
//!
//! The four constraint forms are the textbook ones:
//!
//! - **base** `o ∈ pts(n)` — an allocation seeds its own object.
//! - **copy** `pts(dst) ⊇ pts(src)` — every copy-shaped flow (assignment, phi, `Select`, …).
//! - **load** `pts(dst) ⊇ pts(*ptr · path)` — for each `o ∈ pts(ptr)`, copy from object cell
//!   `Obj(o, path)` into `dst`.
//! - **store** `pts(*ptr · path) ⊇ pts(src)` — for each `o ∈ pts(ptr)`, copy `src` into
//!   `Obj(o, path)`.
//!
//! Multi-level structure (nested refs, array cells) is handled entirely by the *builder* emitting
//! one constraint per type level; the solver stays single-level and oblivious to types.
//!
//! ## Performance
//!
//! This solver uses Andersen-style difference propagation for improved performance over the naïve
//! re-flow of points-to sets. Alongside each node's committed set `sol(n)` it keeps a *delta*
//! `diff(n)` — the objects added to `sol(n)` that have not yet been propagated. A dequeue drains
//! `diff(n)` and acts on only those objects: it unions them into each copy-edge successor, and for
//! each load/store keyed on `n` it wires a cell edge for each *newly discovered* pointee.
//! Already-seen objects are skipped — their edges were wired and their values propagated on an
//! earlier visit.
//!
//! The one exception is a *newly added* copy edge `src → dst` (created while resolving a load or
//! store): `dst` has seen none of `src`'s objects yet, so the full `sol(src)` flows along it once,
//! immediately; from then on only deltas travel the edge.
//!
//! This reaches the same least fixpoint — each object is propagated out of a given node at most
//! once, so total work is bounded by Σ|sol(n)| plus the edges rather than a per-requeue re-scan.
//! The `edges` set additionally dedups copy edges so none is added twice.

use std::collections::VecDeque;

use crate::{
    collections::{HashMap, HashSet},
    compiler::analysis::points_to::object::{AbstractObject, NodeKey, Path},
};

// CONSTRAINT SET
// ================================================================================================

/// A bag of points-to constraints, accumulated by the builder and handed to
/// [`ConstraintSet::solve`].
#[derive(Clone, Debug, Default)]
pub struct ConstraintSet {
    base: Vec<(NodeKey, AbstractObject)>,

    /// `(dst, src)` meaning `pts(dst) ⊇ pts(src)`.
    copies: Vec<(NodeKey, NodeKey)>,

    /// `(dst, ptr, path)` meaning `pts(dst) ⊇ pts(Obj(o, path))` for every `o ∈ pts(ptr)`.
    loads: Vec<(NodeKey, NodeKey, Path)>,

    /// `(ptr, path, src)` meaning `pts(Obj(o, path)) ⊇ pts(src)` for every `o ∈ pts(ptr)`.
    stores: Vec<(NodeKey, Path, NodeKey)>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// `o ∈ pts(node)`.
    pub fn add_base(&mut self, node: NodeKey, obj: AbstractObject) {
        self.base.push((node, obj));
    }

    /// `pts(dst) ⊇ pts(src)`.
    pub fn add_copy(&mut self, dst: NodeKey, src: NodeKey) {
        if dst != src {
            self.copies.push((dst, src));
        }
    }

    /// `pts(dst) ⊇ pts(*ptr · path)`.
    pub fn add_load(&mut self, dst: NodeKey, ptr: NodeKey, path: Path) {
        self.loads.push((dst, ptr, path));
    }

    /// `pts(*ptr · path) ⊇ pts(src)`.
    pub fn add_store(&mut self, ptr: NodeKey, path: Path, src: NodeKey) {
        self.stores.push((ptr, path, src));
    }

    /// Solve to the least fixpoint.
    pub fn solve(self) -> PointsToSolution {
        Solver::new(self).run()
    }
}

// SOLUTION
// ================================================================================================

/// The solved points-to relation: every node mapped to the set of objects it may point to.
#[derive(Clone, Debug, Default)]
pub struct PointsToSolution {
    pts: HashMap<NodeKey, HashSet<AbstractObject>>,
}

impl PointsToSolution {
    /// The points-to set of `node` (empty if `node` points nowhere).
    pub fn get(&self, node: &NodeKey) -> &HashSet<AbstractObject> {
        static EMPTY: std::sync::OnceLock<HashSet<AbstractObject>> = std::sync::OnceLock::new();
        self.pts
            .get(node)
            .unwrap_or_else(|| EMPTY.get_or_init(HashSet::default))
    }

    /// All `(node, pts-set)` pairs (for debugging / annotation).
    pub fn iter(&self) -> impl Iterator<Item = (&NodeKey, &HashSet<AbstractObject>)> {
        self.pts.iter()
    }
}

// SOLVER
// ================================================================================================

struct Solver {
    /// The committed (total) points-to set of each node — the solution returned at the end.
    sol: HashMap<NodeKey, HashSet<AbstractObject>>,

    /// Per-node *delta*: objects added to `sol[n]` but not yet propagated along `n`'s copy edges or
    /// resolved against `n`'s loads/stores. A dequeue drains this (`mem::take`) and acts only on
    /// it.
    diff: HashMap<NodeKey, HashSet<AbstractObject>>,

    /// Copy edges keyed by source: `succ[src]` lists every `dst` with `pts(dst) ⊇ pts(src)`.
    succ: HashMap<NodeKey, Vec<NodeKey>>,

    /// Copy edges already present, to dedup the dynamically-added ones from load/store resolution.
    edges: HashSet<(NodeKey, NodeKey)>,

    /// Loads keyed by pointer node: `loads_by_ptr[ptr]` lists `(dst, path)`.
    loads_by_ptr: HashMap<NodeKey, Vec<(NodeKey, Path)>>,

    /// Stores keyed by pointer node: `stores_by_ptr[ptr]` lists `(src, path)`.
    stores_by_ptr: HashMap<NodeKey, Vec<(NodeKey, Path)>>,

    /// The worklist of nodes that still need to be revisited.
    worklist: VecDeque<NodeKey>,

    /// The nodes that already exist in the queue and hence should not be re-queued (purely a
    /// performance optimization).
    in_queue: HashSet<NodeKey>,
}

impl Solver {
    fn new(cs: ConstraintSet) -> Self {
        let mut s = Solver {
            sol: HashMap::default(),
            diff: HashMap::default(),
            succ: HashMap::default(),
            edges: HashSet::default(),
            loads_by_ptr: HashMap::default(),
            stores_by_ptr: HashMap::default(),
            worklist: VecDeque::new(),
            in_queue: HashSet::default(),
        };

        for (dst, src) in cs.copies {
            s.add_edge(src, dst);
        }
        for (dst, ptr, path) in cs.loads {
            s.loads_by_ptr.entry(ptr).or_default().push((dst, path));
        }
        for (ptr, path, src) in cs.stores {
            s.stores_by_ptr.entry(ptr).or_default().push((src, path));
        }

        // Seed base facts last so the nodes they touch are queued for the first sweep. A base
        // object enters both `sol` and the node's initial `diff`, so the first dequeue propagates
        // it.
        for (node, obj) in cs.base {
            if s.sol.entry(node.clone()).or_default().insert(obj.clone()) {
                s.diff.entry(node.clone()).or_default().insert(obj);
                s.enqueue(node);
            }
        }
        s
    }

    /// Record a copy edge `src → dst` (`pts(dst) ⊇ pts(src)`); returns whether it was new.
    fn add_edge(&mut self, src: NodeKey, dst: NodeKey) -> bool {
        if src == dst || !self.edges.insert((src.clone(), dst.clone())) {
            return false;
        }
        self.succ.entry(src).or_default().push(dst);
        true
    }

    fn enqueue(&mut self, node: NodeKey) {
        if self.in_queue.insert(node.clone()) {
            self.worklist.push_back(node);
        }
    }

    /// Union `objs` into `sol(dst)`. Objects that are *new* to `dst` are recorded in `diff(dst)`
    /// (so a later dequeue propagates them onward) and `dst` is enqueued.
    fn flow(&mut self, dst: &NodeKey, objs: &HashSet<AbstractObject>) {
        if objs.is_empty() {
            return;
        }
        // Collect the newly-inserted objects first, so the `sol` borrow ends before touching
        // `diff`.
        let set = self.sol.entry(dst.clone()).or_default();
        let mut added: Vec<AbstractObject> = Vec::new();
        for o in objs {
            if set.insert(o.clone()) {
                added.push(o.clone());
            }
        }
        if added.is_empty() {
            return;
        }
        self.diff.entry(dst.clone()).or_default().extend(added);
        self.enqueue(dst.clone());
    }

    fn run(mut self) -> PointsToSolution {
        while let Some(n) = self.worklist.pop_front() {
            self.in_queue.remove(&n);

            // Difference propagation: act only on the objects added to `n` since it was last
            // visited, draining its delta. Old objects were already propagated and already wired
            // their load/store cell-edges, so reprocessing them would be pure waste.
            let delta = match self.diff.get_mut(&n) {
                Some(d) if !d.is_empty() => std::mem::take(d),
                _ => continue,
            };

            // Propagate the delta along existing copy edges out of `n`.
            if let Some(dsts) = self.succ.get(&n).cloned() {
                for dst in dsts {
                    self.flow(&dst, &delta);
                }
            }

            // Resolve loads `dst ⊇ *(n · path)` for the *newly discovered* pointees in the delta:
            // each wires a cell→dst copy edge (the full cell set flows once when the edge is new).
            if let Some(entries) = self.loads_by_ptr.get(&n).cloned() {
                for (dst, path) in entries {
                    for o in &delta {
                        // A load through an opaque object (`External`/`Global`) may yield anything
                        // as it denotes unknown memory with no analyzable contents — so it yields
                        // `External` rather than the (empty) opaque cell. Without this a load
                        // through such a ref would under-approximate to ∅.
                        if o.is_opaque() {
                            self.flow(&dst, external_singleton());
                            continue;
                        }

                        let cell = NodeKey::Obj(o.clone(), path.clone());
                        if self.add_edge(cell.clone(), dst.clone()) {
                            let cell_objs = self.sol.get(&cell).cloned().unwrap_or_default();
                            self.flow(&dst, &cell_objs);
                        }
                    }
                }
            }

            // Resolve stores `*(n · path) ⊇ src` for the newly discovered pointees: each wires a
            // src→cell copy edge (the full src set flows once when the edge is new).
            if let Some(entries) = self.stores_by_ptr.get(&n).cloned() {
                for (src, path) in entries {
                    for o in &delta {
                        let cell = NodeKey::Obj(o.clone(), path.clone());
                        if self.add_edge(src.clone(), cell.clone()) {
                            let src_objs = self.sol.get(&src).cloned().unwrap_or_default();
                            self.flow(&cell, &src_objs);
                        }
                    }
                }
            }
        }

        PointsToSolution { pts: self.sol }
    }
}

/// The shared singleton `{External}` set used when a load resolves through an opaque object.
fn external_singleton() -> &'static HashSet<AbstractObject> {
    static EXTERNAL: std::sync::OnceLock<HashSet<AbstractObject>> = std::sync::OnceLock::new();
    EXTERNAL.get_or_init(|| std::iter::once(AbstractObject::External).collect())
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::{
            points_to::object::{Descent, Owner},
            shared::call_string::Context,
        },
        ssa::{FunctionId, ValueId},
    };

    fn f() -> FunctionId {
        FunctionId(0)
    }

    fn alloc(v: u64) -> AbstractObject {
        AbstractObject::Alloc(f(), ValueId(v), Context::empty())
    }

    fn val(v: u64) -> NodeKey {
        NodeKey::value(ValueId(v))
    }

    fn objs(o: &[AbstractObject]) -> HashSet<AbstractObject> {
        o.iter().cloned().collect()
    }

    /// `p = &a` then `q = p` (copy): both point to `a`.
    #[test]
    fn copy_propagates_objects() {
        let mut cs = ConstraintSet::new();
        cs.add_base(val(1), alloc(1)); // p = &a1
        cs.add_copy(val(2), val(1)); // q ⊇ p
        let sol = cs.solve();
        assert_eq!(*sol.get(&val(1)), objs(&[alloc(1)]));
        assert_eq!(*sol.get(&val(2)), objs(&[alloc(1)]));
    }

    /// Copy is one-directional: `q ⊇ p` does not make `p ⊇ q`.
    #[test]
    fn copy_is_directional() {
        let mut cs = ConstraintSet::new();
        cs.add_base(val(2), alloc(2)); // q = &a2
        cs.add_copy(val(2), val(1)); // q ⊇ p  (p still empty)
        let sol = cs.solve();
        assert!(sol.get(&val(1)).is_empty());
        assert_eq!(*sol.get(&val(2)), objs(&[alloc(2)]));
    }

    /// The Andersen precision win: two distinct objects joined into one value's set stay distinct
    /// objects — a `Select`/phi unions, it does not weld. `pts(merged) = {a1, a2}` while
    /// `pts(ra) ∩ pts(rb) = ∅`.
    #[test]
    fn select_unions_without_welding() {
        let mut cs = ConstraintSet::new();
        cs.add_base(val(1), alloc(1)); // ra = &a1
        cs.add_base(val(2), alloc(2)); // rb = &a2
        cs.add_copy(val(3), val(1)); // merged ⊇ ra
        cs.add_copy(val(3), val(2)); // merged ⊇ rb
        let sol = cs.solve();
        assert_eq!(*sol.get(&val(3)), objs(&[alloc(1), alloc(2)]));
        // ra and rb do not alias: their points-to sets are disjoint.
        let ra = sol.get(&val(1));
        let rb = sol.get(&val(2));
        assert!(ra.is_disjoint(rb));
    }

    /// `*p = q; r = *p` forwards `q`'s object through the heap cell of `p`'s pointee.
    #[test]
    fn store_then_load_forwards_through_cell() {
        let mut cs = ConstraintSet::new();
        // p = &a1 ; q = &a2 ; *p = q ; r = *p
        cs.add_base(val(1), alloc(1));
        cs.add_base(val(2), alloc(2));
        cs.add_store(val(1), vec![], val(2)); // *p ⊇ q
        cs.add_load(val(3), val(1), vec![]); // r ⊇ *p
        let sol = cs.solve();
        // r should point to a2, the object stored through p's cell.
        assert_eq!(*sol.get(&val(3)), objs(&[alloc(2)]));
        // and the cell itself records it.
        let cell = NodeKey::Obj(alloc(1), vec![]);
        assert_eq!(*sol.get(&cell), objs(&[alloc(2)]));
    }

    /// Load resolution must add edges for objects discovered *after* the load is first seen: here
    /// `p`'s pointee set grows (via a later copy) and the load must pick the new object up.
    #[test]
    fn load_picks_up_later_discovered_pointees() {
        let mut cs = ConstraintSet::new();
        cs.add_base(val(1), alloc(1)); // p = &a1
        cs.add_load(val(2), val(1), vec![]); // r ⊇ *p   (before *p has contents)
        cs.add_base(val(3), alloc(3)); // s = &a3
        cs.add_store(val(1), vec![], val(3)); // *p ⊇ s   (fills the cell afterwards)
        let sol = cs.solve();
        assert_eq!(*sol.get(&val(2)), objs(&[alloc(3)]));
    }

    /// Field-sensitive array cells: a store into constant cell 0 and a load of constant cell 1 do
    /// not interfere, because they key on distinct `Obj(o, Elem(Index(k)))` nodes.
    #[test]
    fn distinct_constant_cells_do_not_interfere() {
        use super::super::object::Cell;
        let arr = alloc(1);
        let cell0 = vec![Descent::Elem(Cell::Index(0))];
        let cell1 = vec![Descent::Elem(Cell::Index(1))];
        let mut cs = ConstraintSet::new();
        cs.add_base(val(1), arr.clone()); // p = &array
        cs.add_base(val(2), alloc(2)); // x = &a2
        cs.add_store(val(1), cell0.clone(), val(2)); // p[0] = x
        cs.add_load(val(3), val(1), cell1.clone()); // r = p[1]
        let sol = cs.solve();
        // r reads cell 1, which was never written: empty, not {a2}.
        assert!(sol.get(&val(3)).is_empty());
        // cell 0 holds a2.
        assert_eq!(
            *sol.get(&NodeKey::Obj(arr, cell0)),
            objs(&[AbstractObject::Alloc(f(), ValueId(2), Context::empty())])
        );
    }

    /// A points-to cycle (`*p` reaches an object whose cell reaches back) terminates at its least
    /// fixpoint rather than spinning.
    #[test]
    fn cyclic_points_to_terminates() {
        let mut cs = ConstraintSet::new();
        // p = &a1 ; *p = p  -> a1's cell points to a1.
        cs.add_base(val(1), alloc(1));
        cs.add_store(val(1), vec![], val(1));
        cs.add_load(val(2), val(1), vec![]); // r = *p
        let sol = cs.solve();
        assert_eq!(*sol.get(&val(2)), objs(&[alloc(1)]));
        assert_eq!(*sol.get(&NodeKey::Obj(alloc(1), vec![])), objs(&[alloc(1)]));
    }

    /// `External` in a pointer's set means may-point-to-anything; it simply rides along like any
    /// other object and never blocks termination.
    #[test]
    fn external_object_propagates() {
        let mut cs = ConstraintSet::new();
        cs.add_base(val(1), AbstractObject::External);
        cs.add_copy(val(2), val(1));
        let sol = cs.solve();
        assert!(sol.get(&val(2)).contains(&AbstractObject::External));
    }

    /// A node that is a formal owner (not a value) flows like any other node.
    #[test]
    fn formal_owners_flow() {
        let param = NodeKey::Val(Owner::Param(0), vec![]);
        let ret = NodeKey::Val(Owner::Return(0), vec![]);
        let mut cs = ConstraintSet::new();
        cs.add_base(param.clone(), alloc(7));
        cs.add_copy(ret.clone(), param);
        let sol = cs.solve();
        assert_eq!(*sol.get(&ret), objs(&[alloc(7)]));
    }
}
