//! An Andersen-style (inclusion-based) points-to analysis over HLSSA.
//!
//! This is a precise, alias-splitting analysis, maintaining a per-object points-to set, so merged
//! but distinct objects remain distinguishable. It is primarily designed as an enabler for
//! **allocation splitting**.
//!
//! ## Structure
//!
//! The structure of this analysis is as follows:
//!
//! - [`object`] is the abstract domain: [`object::AbstractObject`], the cell-refined
//!   [`object::Descent`] / [`object::Path`] skeleton, and the solver [`object::NodeKey`] universe.
//! - [`solver`] is the inclusion-constraint set and its worklist fixpoint solver (set-valued, with
//!   load/store edges added dynamically as points-to sets grow).
//! - `builder` is the per-opcode constraint walker.
//! - `summary` is the two-phase context-sensitive interprocedural layer (parametric per-function
//!   summaries + a per-context BFS), so calls/returns/arg-out resolve precisely across boundaries.
//! - `array_cells` is the flow-group pre-pass that drives per-constant-index (`Index(k)`) array-cell
//!   precision and the [`PointsTo::splittable_cells`] query.
//!
//! ## Correctness
//!
//! The analysis is **sound** (a may-analysis). It over-approximates the real points-to relation —
//! `pts` only ever grows to a least fixpoint, so two values are reported as possibly-aliasing whenever
//! they might be, and an object is reported as escaping whenever it might. The splitting transform
//! that consumes it only acts when `pts` _proves_ a safety property, so imprecision only ever
//! blocks an optimization, never enables an unsound one. Opaque objects (`External`/`Global`) are
//! handled soundly on both the load side (a load through one yields `External`) and the store side
//! (anything stored through one escapes).
//!
//! The analysis is **complete** in the sense of being _exact for its abstraction_. The solver
//! computes the _least_ solution of the emitted inclusion constraints, so no value's points-to set
//! contains an object that some constraint chain does not force — there is no spurious aliasing
//! beyond what the abstraction itself requires. The interprocedural layer is faithful, not lossy: a
//! function's summary is its exact transfer function over the placeholder inputs (returned objects,
//! per-cell arg-out writes, and leaks), so instantiating it at a call reproduces what inlining the
//! callee with the caller's actual argument objects would yield, and it is monotone in the callee
//! summary (a larger summary yields a correspondingly larger caller solution, never a smaller one).
//!
//! The precision the analysis does _not_ reach is exactly the set of deliberate abstraction choices
//! listed under "Future Extensions & Deliberate Limitations" below.
//!
//! The analysis is **total**. The [`builder`]'s per-opcode walker matches every HLSSA `OpCode`
//! exhaustively. Opcodes that cannot occur at this pipeline stage are rejected loudly rather than
//! silently mishandled — tuples ICE via `ice_non_elided_tuple` (they are elided upstream), and
//! witness/R1CS-era opcodes (`FreshWitness`, `Constrain`, `BumpD`, `Lookup`, …) and dynamic call
//! targets ICE (they appear only after this pass). Every remaining opcode either contributes
//! pointer-flow constraints or is explicitly a no-op; the [`array_cells`] pre-pass and summary
//! extraction likewise cover every relevant opcode (an unrecognised opcode there defaults soundly
//! to a collapsed group). The query surface is likewise total: `points_to`/`may_alias` and their
//! `*_in` variants return the empty set / `false` for an absent value, function, or context rather
//! than panicking.
//!
//! The analysis **terminates** at every layer, each over a finite domain. Both the object and node
//! universes are finite, with `Elem(AllElems)` collapsing dynamically-indexed arrays to a single
//! cell. The solver's points-to sets only ever grow by union and its dynamically-added load/store
//! copy-edges are deduplicated and bounded by the finite node set, so the worklist drains by the
//! ascending-chain condition (the opaque-load special-case adds `External` at most once and creates
//! no opaque object-cell edges).
//!
//! The Phase-1 summary fixpoint converges because summaries grow monotonically over the finite
//! summary lattice and a callee is re-queued only when its summary grows; recursion folds (a
//! recursive call's self-substitution is a self-copy the solver drops). The Phase-2 context BFS
//! visits each `(FunctionId, Context)` once over the finite k-limited context space (a recursive
//! call's pushed context repeats and is not re-enqueued), and the [`array_cells`] pre-pass is two
//! linear passes plus a path-halving union-find.
//!
//! ## Future Extensions & Deliberate Limitations
//!
//! Several precision refinements are intentionally _not_ implemented in the current form of this
//! analysis as they are low ROI for mem2reg and SROA-splitting transforms. Each area mentioned
//! below is handled by a sound over-approximation today. While this blocks some splits, it never
//! enables an unsound one and hence should only be revisited if there is a need.
//!
//! - **Flow-Insensitivity:** The current analysis only exhibits weak updates, storing a union into
//!   the pointee cell. The SROA transform supplies its own flow-sensitivity (a la Cytron mem2reg)
//!   and hence strong updates (overwrite/kill) here would only sharpen `may_alias` in the rare case
//!   of reused mutable nested-ref slots at the cost of a significantly heavier analysis.
//! - **Opaque Source Precision:** Unconstrained call results stay `External` as their input
//!   dependence re-enters structurally as `WriteWitness` rather than splittable heap refs. `Global`
//!   objects carry per-slot identity (so a global read does not alias everything) but not per-slot
//!   *contents*. A future whole-program global-constants pre-pass (`InitGlobal` → `Global` cells
//!   seeded into every `ReadGlobal`) would tighten loads through global-derived refs. As Globals
//!   are init-time constants and program-wide escaped this rarely unblocks a split.
//! - **Context Depth (1-CFA):** The context-sensitive layer is currently `k = 1` as the splitting
//!   decision is context-independent and uses the join over the contexts. This means that deeper
//!   `k` would only sharpen the per-context `*_in` and `may_alias` queries, while risking call
//!   string blowup. As these are never issued by the planned clients of this analysis, we leave it
//!   out for now.
//! - **Nested Arrays:** Only a value's _top_ array level gets per-`Index(k)` cells, while deeper
//!   levels soundly collapse to `Elem(AllElems)`. Full nesting would need an even more complex
//!   path-indexed multi-level flow domain with a 2-D wholesale-copy alignment (a per-outer-cell
//!   inner index set that must stay aligned across `ArraySet`/phi boundaries as misalignment here
//!   is quietly unsound). This would result in quadratic `ArraySet` handling for near zero-payoff
//!   as we claim that ZK nested arrays are overwhelmingly loop-indexed (and hence dynamic and
//!   already collapsed).
//! - **Slices:** Dynamic-length slices never get per-`Index(k)` cells — every slice value and
//!   `SlicePush` result collapses to `Elem(AllElems)`. Constant-cell enumeration is ill-defined for
//!   a runtime-sized container, so this is effectively forced rather than a tunable trade-off.
//! - **All-or-Nothing Group Collapse:** A single collapse trigger (one dynamic index, a
//!   `MkRepeated` of a non-scalar element, a slice op) collapses an array's *entire* flow-group to
//!   one `AllElems` cell,
//!   rather than keeping its constant `Index(k)` cells alongside an `AllElems` overflow (the
//!   textbook field-sensitive model: a dynamic write weak-updates `AllElems`; a constant read sees
//!   `Index(k) ∪ AllElems`). The hybrid would only sharpen `may_alias` on groups mixing constant
//!   and dynamic accesses, and such a group is never `splittable_cells` (splitting needs *every*
//!   access constant), so the planned SROA/mem2reg clients gain nothing — hence the simpler
//!   disjoint-roles invariant (`Split` xor `Collapsed`).

mod array_cells;
mod builder;
pub mod object;
pub mod solver;
mod summary;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis, points_to::builder::build_function, types::TypeInfo,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            FunctionId, Instruction, Terminator, ValueId,
            hlssa::{HLFunction, HLSSA, OpCode},
        },
    },
};

use array_cells::ArrayCells;
use object::{AbstractObject, Context, NodeKey};
use solver::PointsToSolution;

// POINTS-TO ANALYSIS RESULT
// ================================================================================================

/// The solved points-to relation for the whole program, one [`PointsToSolution`] per function.
///
/// Calls are resolved interprocedurally by **context-sensitive summaries** (see [`summary`]). A
/// `Call` instantiates the callee's [`summary::PointsToSummary`], so a ref passed to a non-leaking
/// callee stays local and a ref returned from a callee resolves to the callee's actual object.
///
/// The per-function solutions stored here are the polymorphic (context-insensitive) view used by
/// the allocation-splitting transform on the pre-monomorphization body; the per-context (Phase 2)
/// refinement is held alongside in [`contexts`](Self::contexts) and queried through the `*_in`
/// methods. The query surface ([`PointsTo::may_alias`], [`PointsTo::escapes`],
/// [`PointsTo::is_singleton_reached`], …) is what that transform consumes.
///
/// How a value is used with respect to a pointer, as reported by
/// [`PointsTo::classify_pointer_uses`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerUse {
    /// A `Load`/`Store` *pointer* operand whose points-to set is ambiguous (more than one possible
    /// object): no object it may touch can be strongly updated or promoted.
    Deref,

    /// A `Store` *pointer* operand (reported regardless of ambiguity): every object it may touch is
    /// written through — the out-direction an in/out parameter must replay.
    Write,

    /// A non-pointer use — a stored value, a non-`Load`/`Store` instruction input, or a terminator
    /// operand — so any object the value may point to is consumed *as a value* and its allocation
    /// cannot be removed.
    Consume,
}

#[derive(Debug)]
pub struct PointsTo {
    /// The polymorphic (context-insensitive) solution per function.
    ///
    /// The default query view and the one the splitting transform consumes on the
    /// pre-monomorphization body (the meet over contexts). Computed with `k = 0`.
    functions: HashMap<FunctionId, PointsToSolution>,

    /// The per-context (`k`-CFA) solutions, keyed `(function, call context)`.
    ///
    /// The context-sensitive refinement queried via the `*_in` methods. `main` appears under
    /// `Context::empty()`.
    contexts: HashMap<(FunctionId, Context), PointsToSolution>,

    /// Program-wide set of objects that reach a true sink (global, `main` return, leaking callee),
    /// unioned across every function's escape closure.
    ///
    /// Context-stripped (escape is context-independent), so `escapes` strips an object's context
    /// before checking.
    escaped: HashSet<AbstractObject>,

    /// Per-function array flow-group classification (for `splittable_cells`).
    ///
    /// A function of body shape only, so computed once per function and shared across both solve
    /// passes.
    array_cells: HashMap<FunctionId, ArrayCells>,
}

impl Analysis for PointsTo {
    fn dependencies() -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }

    fn compute(ssa: &HLSSA, store: &AnalysisStore) -> Self {
        PointsTo::run(ssa, store.get::<FlowAnalysis>(), store.get::<TypeInfo>())
    }
}

impl PointsTo {
    /// Solve points-to for every function in `ssa`, resolving calls against the interprocedural
    /// summaries.
    pub fn run(ssa: &HLSSA, flow: &FlowAnalysis, types: &TypeInfo) -> Self {
        /// Call-string depth for the context-sensitive (Phase 2) pass — 1-CFA.
        const K: usize = 1;

        let main = ssa.get_unique_entrypoint_id();

        // Array flow-group classification — a function of body shape only, so computed once per
        // function and reused across both the summary fixpoint and the per-context BFS.
        let array_cells: HashMap<FunctionId, ArrayCells> = ssa
            .get_function_ids()
            .filter(|f| types.has_function(*f))
            .map(|f| {
                let cells = ArrayCells::classify(ssa, ssa.get_function(f), types.get_function(f));
                (f, cells)
            })
            .collect();

        let summaries = summary::compute_summaries(ssa, flow, types, &array_cells);

        // Polymorphic (context-insensitive) per-function solutions + the program-wide escape set.
        // This re-builds and re-solves each function once beyond the summary fixpoint (which keeps
        // only summaries, discarding their solutions). Memoizing the fixpoint's final visit is
        // fiddly — the last visit of a function isn't known until the worklist drains — so the
        // extra pass is an accepted cost.
        let mut functions = HashMap::default();
        let mut escaped: HashSet<AbstractObject> = HashSet::default();
        for fid in ssa.get_function_ids() {
            if !types.has_function(fid) {
                continue;
            }
            let fc = build_function(
                ssa,
                ssa.get_function(fid),
                types.get_function(fid),
                &array_cells[&fid],
                fid,
                &Context::empty(),
                &summaries,
                fid == main,
                0,
            );
            let solution = fc.constraints.solve();
            escaped.extend(compute_escaped(&solution, &fc.escape_roots));
            functions.insert(fid, solution);
        }

        // Context-sensitive per-context solutions (Phase 2).
        let contexts = summary::specialize(ssa, types, &summaries, K, &array_cells);

        PointsTo {
            functions,
            contexts,
            escaped,
            array_cells,
        }
    }

    /// The set of objects the SSA value `v` (in function `f`) may point to.
    ///
    /// Returns the empty set if `f` was not analyzed (e.g. it lacks type info) or `v` points
    /// nowhere — symmetric with [`Self::points_to_in`], so no query panics on an absent function.
    pub fn points_to(&self, f: FunctionId, v: ValueId) -> &HashSet<AbstractObject> {
        match self.functions.get(&f) {
            Some(sol) => sol.get(&NodeKey::value(v)),
            None => empty_object_set(),
        }
    }

    /// Whether `a` and `b` (in function `f`) may point to the same object.
    ///
    /// `true` if their points-to sets intersect or either may point anywhere (`External`).
    pub fn may_alias(&self, f: FunctionId, a: ValueId, b: ValueId) -> bool {
        let pa = self.points_to(f, a);
        let pb = self.points_to(f, b);
        pa.contains(&AbstractObject::External)
            || pb.contains(&AbstractObject::External)
            || !pa.is_disjoint(pb)
    }

    /// Whether object `o` escapes (reaches a global, a `main` return, or a leaking callee — or is
    /// inherently opaque).
    ///
    /// Escape is context-independent, so `o`'s context is stripped before the (context-insensitive)
    /// escape set is consulted.
    pub fn escapes(&self, o: &AbstractObject) -> bool {
        o.is_inherently_escaped() || self.escaped.contains(&strip_context(o))
    }

    /// Whether `v` is reached by exactly one concrete, non-escaping object.
    ///
    /// This is the precondition for being able to promote its allocation to SSA values. The caller
    /// still must confirm that every access goes through `v`'s own pointer; this is only the
    /// points-to half of that test.
    pub fn is_singleton_reached(&self, f: FunctionId, v: ValueId) -> bool {
        let pts = self.points_to(f, v);
        if pts.len() != 1 {
            return false;
        }
        let only = pts.iter().next().expect("len checked");
        !only.is_inherently_escaped() && !self.escapes(only)
    }

    /// The constant cells of the array value `v` (in function `f`) that the SROA splitting
    /// transform may peel into their own values/allocations: `Some(indices)` iff `v`'s flow-group
    /// is `Split`.
    ///
    /// No escape check is needed: a `Split` group is, by construction, never observed as an
    /// aggregate outside its constant-index accesses — every way an array escapes (a `Call`
    /// arg/result, a `Return`, a `Ref<Array>` store/load) is a *collapse trigger*, so it cannot be
    /// `Split`. The cell *contents* escaping (a peeled ref later returned) does not block peeling
    /// as the peeled slot just holds that ref. Works uniformly for scalar and ref arrays.
    pub fn splittable_cells(&self, f: FunctionId, v: ValueId) -> Option<HashSet<usize>> {
        self.array_cells.get(&f)?.split_indices(v).cloned()
    }

    /// Whether `v`'s flow-group in `f` is `Split` (i.e. SROA may peel it). Borrowing and
    /// allocation-free: prefer this over `splittable_cells(..).is_some()` on hot paths, which
    /// clones the index set only to drop it.
    pub fn is_split(&self, f: FunctionId, v: ValueId) -> bool {
        self.array_cells
            .get(&f)
            .map_or(false, |ac| ac.split_indices(v).is_some())
    }

    /// Walk function `f`'s body and report every pointer use, paired with the points-to set at that
    /// site (see [`PointerUse`]).
    ///
    /// This is the shared core of the "is this object used *only* as an unambiguous `Load`/`Store`
    /// pointer?" predicate that both `mem2reg` (which local allocations are promotable) and
    /// `arg_promotion` (which ref parameters are clean, and whether they are written) build on.
    /// Each caller maps the reported set onto its own candidate representation — `mem2reg` keys
    /// candidates by local `Alloc`, `arg_promotion` by parameter placeholder object — so the walk
    /// and the notion of a "disqualifying use" live here once instead of being mirrored across the
    /// two passes (where a fix to one could silently miss the other).
    pub fn classify_pointer_uses(
        &self,
        f: FunctionId,
        func: &HLFunction,
        mut visit: impl FnMut(PointerUse, &HashSet<AbstractObject>),
    ) {
        for (_, block) in func.get_blocks() {
            for instr in block.get_instructions() {
                match instr {
                    // A Load's pointer is a legitimate use; only an *ambiguous* deref disqualifies.
                    OpCode::Load { ptr, .. } => {
                        let pts = self.points_to(f, *ptr);
                        if pts.len() != 1 {
                            visit(PointerUse::Deref, pts);
                        }
                    }
                    OpCode::Store { ptr, value } => {
                        let pts = self.points_to(f, *ptr);
                        if pts.len() != 1 {
                            visit(PointerUse::Deref, pts);
                        }
                        visit(PointerUse::Write, pts);
                        // The stored *value* is a non-pointer use.
                        visit(PointerUse::Consume, self.points_to(f, *value));
                    }
                    // An Alloc defines a ref but uses none.
                    OpCode::Alloc { .. } => {}
                    // Every input of any other opcode is a non-pointer use.
                    other => {
                        for v in other.get_inputs() {
                            visit(PointerUse::Consume, self.points_to(f, *v));
                        }
                    }
                }
            }
            // Terminator operands are non-pointer uses too (a returned/branched ref is consumed).
            match block.get_terminator() {
                Some(Terminator::Return(vals)) => {
                    for v in vals {
                        visit(PointerUse::Consume, self.points_to(f, *v));
                    }
                }
                Some(Terminator::Jmp(_, params)) => {
                    for v in params {
                        visit(PointerUse::Consume, self.points_to(f, *v));
                    }
                }
                Some(Terminator::JmpIf(cond, _, _)) => {
                    visit(PointerUse::Consume, self.points_to(f, *cond))
                }
                None => {}
            }
        }
    }

    // CONTEXT-SENSITIVE QUERIES
    // --------------------------------------------------------------------------------------------
    //
    // The `*_in` queries consult the per-context (k-CFA) solution for `(f, ctx)`, where two call
    // sites of a helper produce distinct context-qualified objects. `main`'s context is
    // `Context::empty()`. A query against a context with no solution returns the empty set / false.

    /// The objects the value `v` may point to in the `(f, ctx)` context.
    pub fn points_to_in(
        &self,
        f: FunctionId,
        ctx: &Context,
        v: ValueId,
    ) -> &HashSet<AbstractObject> {
        match self.contexts.get(&(f, ctx.clone())) {
            Some(sol) => sol.get(&NodeKey::value(v)),
            None => empty_object_set(),
        }
    }

    /// Whether `a` and `b` may alias in the `(f, ctx)` context.
    pub fn may_alias_in(&self, f: FunctionId, ctx: &Context, a: ValueId, b: ValueId) -> bool {
        let pa = self.points_to_in(f, ctx, a);
        let pb = self.points_to_in(f, ctx, b);
        pa.contains(&AbstractObject::External)
            || pb.contains(&AbstractObject::External)
            || !pa.is_disjoint(pb)
    }

    /// Whether `v` is reached by exactly one concrete, non-escaping object in the `(f, ctx)`
    /// context.
    pub fn is_singleton_reached_in(&self, f: FunctionId, ctx: &Context, v: ValueId) -> bool {
        let pts = self.points_to_in(f, ctx, v);
        if pts.len() != 1 {
            return false;
        }
        let only = pts.iter().next().expect("len checked");
        !only.is_inherently_escaped() && !self.escapes(only)
    }
}

// ESCAPE COMPUTATION
// ================================================================================================

/// Close the escape relation.
///
/// Every object reachable from an escape root — directly or through the cells of an already-escaped
/// object — escapes.
pub(super) fn compute_escaped(
    sol: &PointsToSolution,
    roots: &[NodeKey],
) -> HashSet<AbstractObject> {
    // Index each object's cell contents once.
    let mut contents: HashMap<AbstractObject, HashSet<AbstractObject>> = HashMap::default();
    for (node, pts) in sol.iter() {
        if let NodeKey::Obj(o, _) = node {
            contents
                .entry(o.clone())
                .or_default()
                .extend(pts.iter().cloned());
        }
    }

    let mut escaped: HashSet<AbstractObject> = HashSet::default();
    let mut worklist: Vec<AbstractObject> = Vec::new();
    for root in roots {
        for o in sol.get(root) {
            if escaped.insert(o.clone()) {
                worklist.push(o.clone());
            }
        }
    }

    // Anything stored *through* an opaque object (`External`/`Global`) is published to unknown
    // memory, so opaque objects are escape roots: their cell contents escape. Their own
    // escaped-ness is `is_inherently_escaped`; this closes the store-through-opaque hole.
    for o in contents.keys() {
        if o.is_opaque() && escaped.insert(o.clone()) {
            worklist.push(o.clone());
        }
    }
    while let Some(o) = worklist.pop() {
        if let Some(reach) = contents.get(&o) {
            // Iterate by reference: `escaped`/`worklist` are distinct from `contents`, so cloning
            // the whole reachable set just to walk it is unnecessary — clone only what newly escapes.
            for c in reach {
                if escaped.insert(c.clone()) {
                    worklist.push(c.clone());
                }
            }
        }
    }
    escaped
}

// UTILITIES
// ================================================================================================

/// An object with its allocation context erased (escape is context-independent).
fn strip_context(o: &AbstractObject) -> AbstractObject {
    match o {
        AbstractObject::Alloc(f, v, _) => AbstractObject::Alloc(*f, *v, Context::empty()),
        other => other.clone(),
    }
}

/// A shared empty points-to set for queries against absent contexts.
fn empty_object_set() -> &'static HashSet<AbstractObject> {
    static EMPTY: std::sync::OnceLock<HashSet<AbstractObject>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(HashSet::default)
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::analysis::{flow_analysis::FlowAnalysis, types::Types};
    use crate::compiler::ssa::hlssa::{
        Blob, Constant, SequenceTargetType, Type,
        builder::{HLEmitter, HLSSABuilder},
    };

    fn idx_set(ks: &[usize]) -> HashSet<usize> {
        ks.iter().copied().collect()
    }

    fn fr(n: u64) -> ark_bn254::Fr {
        ark_bn254::Fr::from(n)
    }

    fn solve(ssa: &HLSSA) -> PointsTo {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        PointsTo::run(ssa, &flow, &types)
    }

    /// The headline precision gain over Steensgaard unification.
    ///
    /// A phi merging two distinct allocations (refs cannot be `Select`ed — the frontend bans that —
    /// so they merge through a block parameter) unions their objects into the merged value's set
    /// but keeps the originals disjoint. `ra` and `rb` are proven **not** to alias even though they
    /// merge — where unification welds them forever.
    #[test]
    fn merged_refs_do_not_alias() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let cond = e.add_parameter(Type::bool());
                let ra = e.alloc(Type::field());
                let rb = e.alloc(Type::field());
                let c = e.field_const(fr(5));
                e.store(ra, c);
                e.store(rb, c);

                // Merge ra/rb through a block-parameter phi (the `if cond { ra } else { rb }`
                // shape).
                let merged = e.build_if_else(
                    cond,
                    vec![Type::field().ref_of()],
                    |_| vec![ra],
                    |_| vec![rb],
                )[0];
                let r = e.load(merged);
                e.terminate_return(vec![r]);
                captured = Some((ra, rb, merged));
            });
        }
        let (ra, rb, merged) = captured.unwrap();
        let pt = solve(&ssa);

        assert_eq!(
            pt.points_to(main_id, merged).len(),
            2,
            "merged unions both objects"
        );
        assert!(
            !pt.may_alias(main_id, ra, rb),
            "distinct objects must not alias"
        );
        assert!(
            pt.may_alias(main_id, ra, merged),
            "ra is one of merged's targets"
        );
        assert!(pt.may_alias(main_id, rb, merged));
    }

    /// Two independent allocations that never merge each point to their own singleton object and do
    /// not alias.
    #[test]
    fn unmerged_refs_stay_separate() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let ra = e.alloc(Type::field());
                let rb = e.alloc(Type::field());
                let c = e.field_const(fr(7));
                e.store(ra, c);
                let r = e.load(rb);
                e.terminate_return(vec![r]);
                captured = Some((ra, rb));
            });
        }
        let (ra, rb) = captured.unwrap();
        let pt = solve(&ssa);
        assert!(!pt.may_alias(main_id, ra, rb));
        assert!(pt.is_singleton_reached(main_id, ra));
        assert!(pt.is_singleton_reached(main_id, rb));
    }

    /// A store-then-load through a `Ref<Ref<Field>>` forwards the stored pointer's object: the
    /// loaded ref aliases the stored ref.
    #[test]
    fn store_load_forwards_pointee_object() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.alloc(Type::field().ref_of()); // p: Ref<Ref<Field>>
                let q = e.alloc(Type::field()); // q: Ref<Field>
                e.store(p, q); // *p = q
                let r = e.load(p); // r = *p  (a Ref<Field>)
                let v = e.load(r);
                e.terminate_return(vec![v]);
                captured = Some((q, r));
            });
        }
        let (q, r) = captured.unwrap();
        let pt = solve(&ssa);
        assert!(
            pt.may_alias(main_id, q, r),
            "loaded ref aliases the stored ref"
        );
        assert_eq!(pt.points_to(main_id, q), pt.points_to(main_id, r));
    }

    /// A returned ref escapes; a sibling allocation that stays local does not — so the local one is
    /// singleton-reached (promotable) while the returned one is not.
    #[test]
    fn returned_ref_escapes_local_does_not() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field().ref_of());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let kept = e.alloc(Type::field());
                let returned = e.alloc(Type::field());
                let c = e.field_const(fr(1));
                e.store(kept, c);
                e.store(returned, c);
                e.terminate_return(vec![returned]);
                captured = Some((kept, returned));
            });
        }
        let (kept, returned) = captured.unwrap();
        let pt = solve(&ssa);

        let kept_obj = pt.points_to(main_id, kept).iter().next().unwrap().clone();
        let ret_obj = pt
            .points_to(main_id, returned)
            .iter()
            .next()
            .unwrap()
            .clone();
        assert!(!pt.escapes(&kept_obj), "local allocation does not escape");
        assert!(pt.escapes(&ret_obj), "returned allocation escapes");
        assert!(pt.is_singleton_reached(main_id, kept));
        assert!(!pt.is_singleton_reached(main_id, returned));
    }

    /// A callee that only reads its ref argument does not leak it.
    ///
    /// The caller's allocation passed in stays non-escaping and singleton-reached. This is the
    /// interprocedural precision win over a intraprocedural model, which would escape every call
    /// argument unconditionally.
    #[test]
    fn non_leaking_callee_keeps_arg_local() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let reader = sb.ssa().add_function("reader".to_string());
            // reader(p: Ref<Field>): let _ = *p; return
            sb.modify_function(reader, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let _v = e.load(p);
                e.terminate_return(vec![]);
            });
            // main(): q = alloc; *q = 3; reader(q); return *q
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let q = e.alloc(Type::field());
                let c = e.field_const(fr(3));
                e.store(q, c);
                e.call(reader, vec![q], 0);
                let r = e.load(q);
                e.terminate_return(vec![r]);
                captured = Some(q);
            });
        }
        let q = captured.unwrap();
        let pt = solve(&ssa);
        let q_obj = pt.points_to(main_id, q).iter().next().unwrap().clone();
        assert!(
            !pt.escapes(&q_obj),
            "a read-only callee must not escape the argument"
        );
        assert!(pt.is_singleton_reached(main_id, q));
    }

    /// A ref returned from a callee resolves to the callee's own allocation (not `External`), so
    /// the caller's binding is singleton-reached — `is_singleton_reached` is true only for a
    /// single, concrete, non-escaping object.
    #[test]
    fn returned_ref_resolves_to_callee_alloc() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let make = sb.ssa().add_function("make_ref".to_string());
            // make_ref() -> Ref<Field>: let a = alloc; return a
            sb.modify_function(make, |b| {
                b.function.add_return_type(Type::field().ref_of());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.alloc(Type::field());
                e.terminate_return(vec![a]);
            });
            // main(): z = make_ref(); return *z
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let z = e.call(make, vec![], 1)[0];
                let r = e.load(z);
                e.terminate_return(vec![r]);
                captured = Some(z);
            });
        }
        let z = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(
            pt.points_to(main_id, z).len(),
            1,
            "z resolves to exactly the callee's allocation, not an opaque set"
        );
        assert!(
            pt.is_singleton_reached(main_id, z),
            "and that object is concrete and non-escaping (not External)"
        );
    }

    /// A callee that stashes its ref argument into a global leaks it: the caller's allocation
    /// passed in therefore escapes (the escape-soundness obligation of dropping unconditional
    /// arg-escape).
    #[test]
    fn callee_stashing_arg_in_global_escapes_it() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().ref_of()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let stash = sb.ssa().add_function("stash".to_string());
            // stash(p: Ref<Field>): global[0] = p; return
            sb.modify_function(stash, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                e.init_global(0, p);
                e.terminate_return(vec![]);
            });
            // main(): q = alloc; *q = 0; stash(q); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let q = e.alloc(Type::field());
                let c = e.field_const(fr(0));
                e.store(q, c);
                e.call(stash, vec![q], 0);
                e.terminate_return(vec![]);
                captured = Some(q);
            });
        }
        let q = captured.unwrap();
        let pt = solve(&ssa);
        let q_obj = pt.points_to(main_id, q).iter().next().unwrap().clone();
        assert!(
            pt.escapes(&q_obj),
            "an argument stashed in a global must escape"
        );
        assert!(!pt.is_singleton_reached(main_id, q));
    }

    /// A self-recursive helper that returns its own ref parameter: the summary fixpoint must
    /// converge (the test terminating is the assertion) and the returned binding must resolve back
    /// to the caller's argument object through the recursion.
    #[test]
    fn recursion_converges_and_returns_resolve() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let rec = sb.ssa().add_function("rec".to_string());
            // rec(p: Ref<Field>) -> Ref<Field>: rec(p); return p
            sb.modify_function(rec, |b| {
                b.function.add_return_type(Type::field().ref_of());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                e.call(rec, vec![p], 1); // self-recursion (result discarded, but arity must match)
                e.terminate_return(vec![p]);
            });
            // main(): q = alloc; z = rec(q); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let q = e.alloc(Type::field());
                let z = e.call(rec, vec![q], 1)[0];
                e.terminate_return(vec![]);
                captured = Some((q, z));
            });
        }
        let (q, z) = captured.unwrap();
        let pt = solve(&ssa);
        assert!(
            pt.may_alias(main_id, z, q),
            "the returned ref resolves back to the caller's argument through the recursion"
        );
        assert!(pt.is_singleton_reached(main_id, q));
    }

    /// Phase 2 (context sensitivity): two call sites of one allocating helper yield distinct
    /// context-qualified objects, so their results do not alias in the per-context view — even
    /// though the polymorphic (context-insensitive) view conflates them.
    #[test]
    fn distinct_call_sites_get_distinct_objects() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let make = sb.ssa().add_function("make".to_string());
            // make() -> Ref<Field>: let a = alloc; return a
            sb.modify_function(make, |b| {
                b.function.add_return_type(Type::field().ref_of());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.alloc(Type::field());
                e.terminate_return(vec![a]);
            });
            // main(): z1 = make(); z2 = make(); let _ = *z1; let _ = *z2; return *z1
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let z1 = e.call(make, vec![], 1)[0];
                let z2 = e.call(make, vec![], 1)[0];
                let r = e.load(z1);
                let _ = e.load(z2);
                e.terminate_return(vec![r]);
                captured = Some((z1, z2));
            });
        }
        let (z1, z2) = captured.unwrap();
        let pt = solve(&ssa);

        // Polymorphic view: both calls conflate to make's single empty-context allocation.
        assert!(
            pt.may_alias(main_id, z1, z2),
            "the context-insensitive view conflates the two call sites"
        );
        // 1-CFA view: distinct call sites ⇒ distinct objects ⇒ no aliasing.
        let entry_ctx = Context::empty();
        assert!(
            !pt.may_alias_in(main_id, &entry_ctx, z1, z2),
            "1-CFA distinguishes the two call sites"
        );
        assert_eq!(pt.points_to_in(main_id, &entry_ctx, z1).len(), 1);
        assert_eq!(pt.points_to_in(main_id, &entry_ctx, z2).len(), 1);
        assert!(pt.is_singleton_reached_in(main_id, &entry_ctx, z1));
        assert!(pt.is_singleton_reached_in(main_id, &entry_ctx, z2));
    }

    // ARRAY-CELL PRECISION
    // --------------------------------------------------------------------------------------------

    /// A purely-local, constant-indexed array of refs is cell-split: cell 0 and cell 1 resolve to
    /// their distinct elements and do not alias, and `splittable_cells` reports both indices.
    #[test]
    fn const_array_cells_split_per_index() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let ra = e.alloc(Type::field());
                let rb = e.alloc(Type::field());
                let arr = e.mk_seq(
                    vec![ra, rb],
                    SequenceTargetType::Array(2),
                    Type::field().ref_of(),
                );
                let i0 = e.u_const(32, 0);
                let i1 = e.u_const(32, 1);
                let r0 = e.array_get(arr, i0);
                let r1 = e.array_get(arr, i1);
                let v = e.load(r0);
                e.terminate_return(vec![v]);
                captured = Some((ra, rb, arr, r0, r1));
            });
        }
        let (ra, rb, arr, r0, r1) = captured.unwrap();
        let pt = solve(&ssa);

        assert_eq!(
            pt.points_to(main_id, r0),
            pt.points_to(main_id, ra),
            "cell 0 → ra"
        );
        assert_eq!(
            pt.points_to(main_id, r1),
            pt.points_to(main_id, rb),
            "cell 1 → rb"
        );
        assert!(
            !pt.may_alias(main_id, r0, r1),
            "distinct const cells do not alias"
        );
        assert_eq!(pt.splittable_cells(main_id, arr), Some(idx_set(&[0, 1])));
    }

    /// A scalar array (no refs) accessed only at constant indices is still reported splittable,
    /// with an empty points-to set — the structural splitting gate the SROA transform consumes.
    #[test]
    fn scalar_array_is_splittable() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c0 = e.field_const(fr(7));
                let c1 = e.field_const(fr(9));
                let arr = e.mk_seq(vec![c0, c1], SequenceTargetType::Array(2), Type::field());
                let i1 = e.u_const(32, 1);
                let got = e.array_get(arr, i1);
                e.terminate_return(vec![got]);
                captured = Some(arr);
            });
        }
        let arr = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(pt.splittable_cells(main_id, arr), Some(idx_set(&[0, 1])));
        assert!(
            pt.points_to(main_id, arr).is_empty(),
            "scalar array has no points-to"
        );
    }

    /// A single dynamic index collapses the whole flow-group: `splittable_cells` is `None` and the
    /// read conservatively sees every element (the union), not one cell.
    #[test]
    fn dynamic_index_collapses_group() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let n = e.add_parameter(Type::u(32)); // dynamic index
                let ra = e.alloc(Type::field());
                let rb = e.alloc(Type::field());
                let arr = e.mk_seq(
                    vec![ra, rb],
                    SequenceTargetType::Array(2),
                    Type::field().ref_of(),
                );
                let r = e.array_get(arr, n); // dynamic ⇒ collapse
                let v = e.load(r);
                e.terminate_return(vec![v]);
                captured = Some((ra, rb, arr, r));
            });
        }
        let (ra, rb, arr, r) = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(
            pt.splittable_cells(main_id, arr),
            None,
            "dynamic index collapses the group"
        );
        // The dynamic read may be either element.
        assert!(pt.may_alias(main_id, r, ra));
        assert!(pt.may_alias(main_id, r, rb));
    }

    /// A phi-merged array stays cell-aligned: cell 0 of the merge sees cell 0 of *both*
    /// predecessors (the wholesale-copy soundness obligation) and never cell 1.
    #[test]
    fn phi_merged_array_is_cell_aligned() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let cond = e.add_parameter(Type::bool());
                let a = e.alloc(Type::field());
                let bb = e.alloc(Type::field());
                let c = e.alloc(Type::field());
                let d = e.alloc(Type::field());
                let arr_t = e.mk_seq(
                    vec![a, bb],
                    SequenceTargetType::Array(2),
                    Type::field().ref_of(),
                );
                let arr_f = e.mk_seq(
                    vec![c, d],
                    SequenceTargetType::Array(2),
                    Type::field().ref_of(),
                );
                let merged = e.build_if_else(
                    cond,
                    vec![Type::field().ref_of().array_of(2)],
                    |_| vec![arr_t],
                    |_| vec![arr_f],
                )[0];
                let i0 = e.u_const(32, 0);
                let r0 = e.array_get(merged, i0);
                let v = e.load(r0);
                e.terminate_return(vec![v]);
                captured = Some((a, bb, c, d, r0));
            });
        }
        let (a, bb, c, d, r0) = captured.unwrap();
        let pt = solve(&ssa);
        // cell 0 of the merge = {a, c} (cell 0 of each predecessor); never b/d (cell 1).
        assert!(pt.may_alias(main_id, r0, a), "sees then-branch cell 0");
        assert!(pt.may_alias(main_id, r0, c), "sees else-branch cell 0");
        assert!(
            !pt.may_alias(main_id, r0, bb),
            "does not see then-branch cell 1"
        );
        assert!(
            !pt.may_alias(main_id, r0, d),
            "does not see else-branch cell 1"
        );
    }

    /// An array stored through a `Ref<Array>` collapses (it enters object-land) and is not
    /// splittable.
    #[test]
    fn array_under_ref_collapses() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let ra = e.alloc(Type::field());
                let rb = e.alloc(Type::field());
                let arr = e.mk_seq(
                    vec![ra, rb],
                    SequenceTargetType::Array(2),
                    Type::field().ref_of(),
                );
                let p = e.alloc(Type::field().ref_of().array_of(2)); // p: Ref<Array<Ref<Field>,2>>
                e.store(p, arr); // arr escapes into object-land ⇒ collapse
                e.terminate_return(vec![]);
                captured = Some(arr);
            });
        }
        let arr = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(
            pt.splittable_cells(main_id, arr),
            None,
            "array under a ref collapses"
        );
    }

    /// A constant-count repeat of a *scalar* element is cell-splittable over `0..count` (the cells
    /// all alias the one source value until an `ArraySet` diverges them — the O1 refinement).
    #[test]
    fn mk_repeated_scalar_is_splittable() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c = e.field_const(fr(3));
                let arr = e.mk_repeated(c, SequenceTargetType::Array(4), 4, Type::field());
                let i0 = e.u_const(32, 0);
                let got = e.array_get(arr, i0);
                e.terminate_return(vec![got]);
                captured = Some(arr);
            });
        }
        let arr = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(
            pt.splittable_cells(main_id, arr),
            Some(idx_set(&[0, 1, 2, 3]))
        );
    }

    /// A repeat of a *ref* element stays collapsed: every cell aliases the one object, so peeling is
    /// pointless and the analysis reports it un-splittable.
    #[test]
    fn mk_repeated_ref_is_not_splittable() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let r = e.alloc(Type::field());
                let c = e.field_const(fr(3));
                e.store(r, c);
                let arr = e.mk_repeated(r, SequenceTargetType::Array(4), 4, Type::field().ref_of());
                let i0 = e.u_const(32, 0);
                let got = e.array_get(arr, i0);
                let v = e.load(got);
                e.terminate_return(vec![v]);
                captured = Some(arr);
            });
        }
        let arr = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(pt.splittable_cells(main_id, arr), None);
    }

    /// A constant blob array, constant-indexed, is cell-splittable per its resolved length (the
    /// `MkSeqOfBlob` refinement — it used to collapse unconditionally).
    #[test]
    fn const_blob_array_is_splittable() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let blob = e.emit_constant(Constant::Blob(Blob {
                    elem_type: Type::field(),
                    elements: vec![
                        Constant::Field(fr(1)),
                        Constant::Field(fr(2)),
                        Constant::Field(fr(3)),
                    ],
                }));
                let arr = e.mk_seq_of_blob(Type::field(), blob);
                let i1 = e.u_const(32, 1);
                let got = e.array_get(arr, i1);
                e.terminate_return(vec![got]);
                captured = Some(arr);
            });
        }
        let arr = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(pt.splittable_cells(main_id, arr), Some(idx_set(&[0, 1, 2])));
    }

    // GLOBALS AND OPACITY
    // --------------------------------------------------------------------------------------------

    /// Two reads of the same global slot point to the same distinct `Global` object, so they alias
    /// — and the object is a single, distinct one (not the may-alias-anything `External`).
    #[test]
    fn two_reads_same_global_alias() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().ref_of()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let r1 = e.read_global(0, Type::field().ref_of());
                let r2 = e.read_global(0, Type::field().ref_of());
                let v = e.load(r1);
                e.terminate_return(vec![v]);
                captured = Some((r1, r2));
            });
        }
        let (r1, r2) = captured.unwrap();
        let pt = solve(&ssa);
        assert!(
            pt.may_alias(main_id, r1, r2),
            "two reads of one global slot alias"
        );
        assert_eq!(
            pt.points_to(main_id, r1).len(),
            1,
            "a single distinct Global object"
        );
    }

    /// A global read does not alias an unrelated local — the anti-virality win over `External`
    /// (which would have forced may-alias-anything).
    #[test]
    fn read_global_does_not_alias_local() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().ref_of()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let r = e.read_global(0, Type::field().ref_of());
                let a = e.alloc(Type::field());
                let v = e.load(a);
                e.terminate_return(vec![v]);
                captured = Some((r, a));
            });
        }
        let (r, a) = captured.unwrap();
        let pt = solve(&ssa);
        assert!(
            !pt.may_alias(main_id, r, a),
            "a global read does not alias an unrelated local"
        );
    }

    /// Opacity soundness (load side): loading *through* a ref read from a global yields `External`
    /// (unknown memory), so the loaded value may-aliases anything — not the unsound empty set.
    #[test]
    fn load_through_global_yields_external() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().ref_of().ref_of()]); // Ref<Ref<Field>>
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let g = e.read_global(0, Type::field().ref_of().ref_of());
                let inner = e.load(g); // Ref<Field>, points to External (load through opaque)
                let a = e.alloc(Type::field());
                let v = e.load(inner);
                e.terminate_return(vec![v]);
                captured = Some((inner, a));
            });
        }
        let (inner, a) = captured.unwrap();
        let pt = solve(&ssa);
        assert!(
            pt.may_alias(main_id, inner, a),
            "load through an opaque global yields External (was unsoundly empty before)"
        );
    }

    /// Opacity soundness (store side): storing a local through a ref read from a global publishes
    /// it to unknown memory, so the stored object escapes.
    #[test]
    fn store_through_opaque_global_escapes() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().ref_of().ref_of()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let g = e.read_global(0, Type::field().ref_of().ref_of());
                let a = e.alloc(Type::field()); // a: Ref<Field>
                e.store(g, a); // *g = a — published through opaque memory
                e.terminate_return(vec![]);
                captured = Some(a);
            });
        }
        let a = captured.unwrap();
        let pt = solve(&ssa);
        let a_obj = pt.points_to(main_id, a).iter().next().unwrap().clone();
        assert!(
            pt.escapes(&a_obj),
            "storing through an opaque global escapes the stored object"
        );
    }

    // PRECISE ARG-OUT
    // --------------------------------------------------------------------------------------------

    /// A callee that allocates and writes a ref *through* a ref parameter: the caller's load of
    /// that param's pointee resolves to the callee's actual allocation — not `External`. The
    /// headline precise-arg-out win over the old `External`-pollution model.
    #[test]
    fn callee_write_through_param_resolves_precisely() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let writer = sb.ssa().add_function("writer".to_string());
            // writer(pp: Ref<Ref<Field>>): let a = alloc; *pp = a; return
            sb.modify_function(writer, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let pp = e.add_parameter(Type::field().ref_of().ref_of());
                let a = e.alloc(Type::field());
                e.store(pp, a);
                e.terminate_return(vec![]);
            });
            // main(): pp = alloc(Ref<Field>); writer(pp); let r = *pp; return *r
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let pp = e.alloc(Type::field().ref_of());
                e.call(writer, vec![pp], 0);
                let r = e.load(pp);
                let v = e.load(r);
                e.terminate_return(vec![v]);
                captured = Some(r);
            });
        }
        let r = captured.unwrap();
        let pt = solve(&ssa);
        assert_eq!(
            pt.points_to(main_id, r).len(),
            1,
            "arg-pointee resolves to the callee's allocation, not the opaque External"
        );
        assert!(
            pt.is_singleton_reached(main_id, r),
            "and that object is concrete and non-escaping (precise, not External-poisoned)"
        );
    }

    /// Placeholder-substitution arg-out: a callee that copies one ref parameter into another's
    /// pointee (`*dst = src`) — at the call, the caller's `*dst` resolves to the caller's `src`
    /// object.
    #[test]
    fn callee_copies_param_into_param_pointee() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut captured = None;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let copy = sb.ssa().add_function("copy".to_string());
            // copy(dst: Ref<Ref<Field>>, src: Ref<Field>): *dst = src; return
            sb.modify_function(copy, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let dst = e.add_parameter(Type::field().ref_of().ref_of());
                let src = e.add_parameter(Type::field().ref_of());
                e.store(dst, src);
                e.terminate_return(vec![]);
            });
            // main(): dst = alloc(Ref<Field>); a = alloc(Field); copy(dst, a); r = *dst
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let dst = e.alloc(Type::field().ref_of());
                let a = e.alloc(Type::field());
                e.call(copy, vec![dst, a], 0);
                let r = e.load(dst);
                let v = e.load(r);
                e.terminate_return(vec![v]);
                captured = Some((r, a));
            });
        }
        let (r, a) = captured.unwrap();
        let pt = solve(&ssa);
        assert!(
            pt.may_alias(main_id, r, a),
            "the param copied through dst resolves to the caller's src object"
        );
        assert_eq!(pt.points_to(main_id, r), pt.points_to(main_id, a));
    }
}
