//! The per-opcode constraint walker for the points-to analysis.
//!
//! For each function it emits the inclusion constraints of every opcode into a [`ConstraintSet`],
//! plus the value nodes whose points-to set *escapes* to a true program sink (a global, or — for
//! `main` — a return). Calls are no longer modeled by blanket escape: a `Call` **instantiates the
//! callee's [`PointsToSummary`]** (see [`super::summary`]), so a ref passed to a callee that does
//! not leak it stays local, and a ref returned from a callee resolves to the callee's actual object
//! rather than `External`.
//!
//! Pointer structure is handled by emitting one constraint per *ref-bearing level* of a type (see
//! [`ref_levels`]): the points-to set lives at each level whose type is a `Ref`, reached by
//! descending only through array elements. Object cells (a `Ref`'s pointee, an array element) are
//! shared through object identity in the solver, never copied — that is the inclusion refinement of
//! the unification `copy_levels`.
//!
//! Array element flow is **cell-sensitive**. The [`array_cells`](super::array_cells) pre-pass
//! classifies each array flow-group as `Split` (purely local, only ever constant-indexed) or
//! `Collapsed`. For a `Split` group the builder emits one `Obj(_, Elem(Index(k)))` node per live
//! constant index, so distinct constant cells carry distinct points-to sets and `may_alias` is
//! answered *per cell*. A `Collapsed` group (any dynamic index, `MkRepeated`, slice, through-`Ref`,
//! or boundary crossing) and every array level below the value's top level collapse to the single
//! [`Cell::AllElems`](super::object::Cell) node. The
//! [`PointsTo::splittable_cells`](super::PointsTo::splittable_cells) query reads the same
//! classification.
//!
//! ## Contexts
//!
//! [`build_function`] takes a creation [`Context`] for *this* function's local allocations and a
//! depth `k`. Phase 1 (summary computation) passes `ctx = empty`, `k = 0`, so every object is
//! polymorphic; Phase 2 passes a per-call-site context and `k ≥ 1`, qualifying local allocations
//! and pushing each call site (`object::Context::push`) so two call sites of one helper get
//! distinct objects.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            points_to::{
                array_cells::ArrayCells,
                object::{AbstractObject, Cell, Context, Descent, NodeKey, Owner, Path},
                solver::ConstraintSet,
                summary::PointsToSummary,
            },
            types::FunctionTypeInfo,
        },
        ssa::{
            FunctionId, Instruction, Terminator, ValueId,
            hlssa::{CallTarget, Constant, HLFunction, HLSSA, OpCode, Type, TypeExpr},
        },
        util::ice_non_elided_tuple,
    },
};

// PUBLIC ENTRY
// ================================================================================================

/// The constraints and escape roots extracted from one function.
pub struct FunctionConstraints {
    /// The constraint set for the function.
    pub constraints: ConstraintSet,

    /// Value nodes whose points-to set reaches a true program sink (a global, a leaking callee
    /// argument, or — for a `main` — a return).
    ///
    /// Every object reachable from one of these roots escapes.
    pub escape_roots: Vec<NodeKey>,

    /// The `(callee, callee_context)` pair for every constrained static call whose summary was
    /// instantiated — the Phase-2 context BFS (`summary::specialize`) walks these.
    ///
    /// Empty in Phase 1 (`k = 0` makes every callee context the empty context).
    pub callee_contexts: Vec<(FunctionId, Context)>,
}

/// Build the points-to constraints for one function in a given creation [`Context`].
///
/// `summaries` supplies the (current) callee summaries instantiated at each `Call`; `is_main` makes
/// returns escape (the program boundary); `k` bounds the call-string depth for context-qualifying
/// callee objects; `cells` classifies array flow-groups for per-`Index(k)` cell precision; `ssa` is
/// used only to resolve constant array indices.
#[allow(clippy::too_many_arguments)]
pub fn build_function(
    ssa: &HLSSA,
    func: &HLFunction,
    types: &FunctionTypeInfo,
    cells: &ArrayCells,
    fid: FunctionId,
    ctx: &Context,
    summaries: &HashMap<FunctionId, PointsToSummary>,
    is_main: bool,
    k: usize,
) -> FunctionConstraints {
    let mut b = FnBuilder {
        ssa,
        types,
        cells,
        fid,
        ctx: ctx.clone(),
        summaries,
        is_main,
        k,
        cs: ConstraintSet::new(),
        escape_roots: Vec::new(),
        callee_contexts: Vec::new(),
    };

    // Formal inputs: each ref-typed parameter level points to a fresh `Placeholder` standing for
    // the caller's memory. Phase 2 substitutes the caller's actual argument objects per context; in
    // Phase 1 the placeholder is the symbolic input the summary is parametric over.
    for (i, (value, ty)) in func.get_entry().get_parameters().enumerate() {
        // The entry function has no caller, so its parameters cannot be ref-typed: a `Placeholder`
        // for a `main` parameter would stand for caller memory that does not exist and is *not*
        // `is_inherently_escaped`, so a local stored through it would be judged non-escaping —
        // unsound. ZK `main` signatures are value-typed, so this never fires; the assert pins the
        // invariant against a future change that would silently under-approximate escape.
        debug_assert!(
            !is_main || ref_levels(ty).is_empty(),
            "entry function must have no ref-typed parameters"
        );
        b.seed_param(*value, i, ty);
    }

    let block_ids: Vec<_> = func.get_blocks().map(|(bid, _)| *bid).collect();
    for bid in &block_ids {
        let block = func.get_block(*bid);
        for instr in block.get_instructions() {
            b.build_instr(instr);
        }
        match block.get_terminator() {
            Some(Terminator::Jmp(target, jump_args)) => {
                let params = func.get_block(*target).get_parameters();
                for ((param, param_type), arg) in params.zip(jump_args.iter()) {
                    b.copy_value(*param, *arg, param_type);
                }
            }
            // JmpIf carries no arguments (merge phis are filled by Jmp predecessors), and the
            // condition is a scalar so there's no pointer flow.
            Some(Terminator::JmpIf(..)) | None => {}
            Some(Terminator::Return(values)) => {
                for (j, value) in values.iter().enumerate() {
                    let return_type = &func.get_returns()[j];

                    // Bind to the `Return(j)` formal so the summary can read what `f` returns.
                    for path in ref_levels(return_type) {
                        b.cs.add_copy(
                            NodeKey::Val(Owner::Return(j), path.clone()),
                            NodeKey::value(*value).extend(&path),
                        );
                    }

                    // A non-main return is NOT a sink — the caller decides whether the returned
                    // object escapes (the summary carries it back). Only `main`'s returns are the
                    // program boundary.
                    if b.is_main {
                        b.escape_value(*value, return_type);
                    }
                }
            }
        }
    }

    FunctionConstraints {
        constraints: b.cs,
        escape_roots: b.escape_roots,
        callee_contexts: b.callee_contexts,
    }
}

// BUILDER
// ================================================================================================

struct FnBuilder<'a> {
    ssa: &'a HLSSA,
    types: &'a FunctionTypeInfo,
    cells: &'a ArrayCells,
    fid: FunctionId,

    /// The creation context for this function's local allocations (empty in Phase 1).
    ctx: Context,

    /// Callee summaries instantiated at each `Call`.
    summaries: &'a HashMap<FunctionId, PointsToSummary>,

    /// Whether this is the entry function (so returns are a true sink).
    is_main: bool,

    /// Call-string depth bound for context-qualifying callee objects (0 in Phase 1).
    k: usize,

    cs: ConstraintSet,
    escape_roots: Vec<NodeKey>,
    callee_contexts: Vec<(FunctionId, Context)>,
}

impl FnBuilder<'_> {
    fn value_type(&self, v: ValueId) -> &Type {
        self.types.get_value_type(v)
    }

    /// Copy the points-to sets of `src` into `dst` at every ref-bearing level of `ty`, splitting
    /// the top array level into `dst`'s flow-group cells (a wholesale array copy must be
    /// cell-aligned).
    ///
    /// `dst` and `src` share a flow-group (they are unioned in the pre-pass), so the cell set
    /// matches.
    fn copy_value(&mut self, dst: ValueId, src: ValueId, ty: &Type) {
        for path in self.value_ref_paths(dst, ty) {
            self.cs.add_copy(
                NodeKey::value(dst).extend(&path),
                NodeKey::value(src).extend(&path),
            );
        }
    }

    /// Copy ref-level points-to from `src_base` into `dst_base` over the shape of `ty` (array
    /// levels collapse to `AllElems` — used for element types and `Collapsed`-group element
    /// copies).
    fn copy_levels(&mut self, dst_base: NodeKey, src_base: NodeKey, ty: &Type) {
        for path in ref_levels(ty) {
            self.cs
                .add_copy(dst_base.extend(&path), src_base.extend(&path));
        }
    }

    /// The element cells of array value `array`: `Split` → one `Index(k)` per live index;
    /// `Collapsed` (or non-array) → the single `AllElems`.
    fn elem_cells(&self, array: ValueId) -> Vec<Cell> {
        match self.cells.split_indices(array) {
            Some(idx) => idx.iter().map(|k| Cell::Index(*k)).collect(),
            None => vec![Cell::AllElems],
        }
    }

    /// The node for one element cell of an array value.
    fn cell_node(&self, array: ValueId, cell: Cell) -> NodeKey {
        NodeKey::value(array).extend(&[Descent::Elem(cell)])
    }

    /// The constant value of `index`, if any.
    fn const_index(&self, index: ValueId) -> Option<usize> {
        match &*self.ssa.get_const(index)? {
            Constant::U(_, v) => Some(*v as usize),
            _ => None,
        }
    }

    /// Ref-bearing paths of a *value* of type `ty`, with the value's own top array level expanded
    /// into its flow-group cells (deeper array levels collapse to `AllElems` — single-level arrays
    /// are the case that matters; nested arrays stay sound).
    ///
    /// Per-inner-cell precision is a deliberate non-goal; see the "Future Extensions & Deliberate
    /// Limitations" section of the module docs.
    fn value_ref_paths(&self, value: ValueId, ty: &Type) -> Vec<Path> {
        let mut out = Vec::new();
        self.collect_value_paths(value, ty, &mut Vec::new(), &mut out, true);
        out
    }

    fn collect_value_paths(
        &self,
        value: ValueId,
        ty: &Type,
        prefix: &mut Path,
        out: &mut Vec<Path>,
        top_array_level: bool,
    ) {
        let ty = ty.peel_witness();
        match &ty.expr {
            TypeExpr::Ref(_) => out.push(prefix.clone()),
            TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
                let cells = if top_array_level {
                    self.elem_cells(value)
                } else {
                    vec![Cell::AllElems]
                };
                for cell in cells {
                    prefix.push(Descent::Elem(cell));
                    self.collect_value_paths(value, inner, prefix, out, false);
                    prefix.pop();
                }
            }
            TypeExpr::Field
            | TypeExpr::U(_)
            | TypeExpr::I(_)
            | TypeExpr::Function
            | TypeExpr::Blob(..) => {}
            TypeExpr::WitnessOf(_) => unreachable!("peeled above"),
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        }
    }

    /// Mark every object reachable through `value`'s ref-levels as escaping.
    fn escape_value(&mut self, value: ValueId, ty: &Type) {
        for path in ref_levels(ty) {
            self.escape_roots.push(NodeKey::value(value).extend(&path));
        }
    }

    /// Seed `External` into every ref-level of `value` (an unresolved/opaque pointer source).
    fn seed_external(&mut self, value: ValueId, ty: &Type) {
        for path in ref_levels(ty) {
            self.cs.add_base(
                NodeKey::value(value).extend(&path),
                AbstractObject::External,
            );
        }
    }

    /// Seed each ref-level of parameter `i` with its `Placeholder` (caller-owned memory).
    fn seed_param(&mut self, value: ValueId, i: usize, ty: &Type) {
        for path in ref_levels(ty) {
            self.cs.add_base(
                NodeKey::value(value).extend(&path),
                AbstractObject::Placeholder(self.fid, i, path),
            );
        }
    }

    /// A canonical leaf value-node whose points-to set is exactly `{o}` ("a reference to `o`"),
    /// used as a store source for a callee's precise arg-out writes.
    ///
    /// Base-seeded idempotently; shared across uses (safe because a concrete object's identity is
    /// call-independent).
    fn obj_ref(&mut self, o: AbstractObject) -> NodeKey {
        let n = NodeKey::ObjRef(o.clone());
        self.cs.add_base(n.clone(), o);
        n
    }

    /// The context callee objects are created in for a call at this site (first result, else first
    /// argument, as a stable per-call id; falls back to the current context for void/no-arg calls).
    fn callee_ctx(&self, args: &[ValueId], results: &[ValueId]) -> Context {
        match results.first().or_else(|| args.first()) {
            Some(v) => self.ctx.push((self.fid, *v), self.k),
            None => self.ctx.clone(),
        }
    }

    fn build_instr(&mut self, instr: &OpCode) {
        match instr {
            // --- Allocation: the one base rule. ---
            OpCode::Alloc { result, .. } => {
                self.cs.add_base(
                    NodeKey::value(*result),
                    AbstractObject::Alloc(self.fid, *result, self.ctx.clone()),
                );
            }

            // --- Copy-shaped value flows. ---
            OpCode::Select {
                result, if_t, if_f, ..
            } => {
                let ty = self.value_type(*result).clone();
                self.copy_value(*result, *if_t, &ty);
                self.copy_value(*result, *if_f, &ty);
            }
            OpCode::Not { result, value }
            | OpCode::Cast { result, value, .. }
            | OpCode::SExt { result, value, .. }
            | OpCode::BitRange { result, value, .. }
            | OpCode::Spread { result, value, .. } => {
                let ty = self.value_type(*result).clone();
                self.copy_value(*result, *value, &ty);
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                ..
            } => {
                // Both halves are scalar bit values — no pointer flow — but mirror the copy for
                // uniformity (ref_levels is empty, so these emit nothing).
                let ty_odd = self.value_type(*result_odd).clone();
                let ty_even = self.value_type(*result_even).clone();
                self.copy_value(*result_odd, *value, &ty_odd);
                self.copy_value(*result_even, *value, &ty_even);
            }

            // --- Memory. ---
            OpCode::Store { ptr, value } => {
                let pointee = self.pointee_of(*ptr, "Store");
                for path in ref_levels(&pointee) {
                    self.cs.add_store(
                        NodeKey::value(*ptr),
                        path.clone(),
                        NodeKey::value(*value).extend(&path),
                    );
                }
            }
            OpCode::Load { result, ptr } => {
                let pointee = self.pointee_of(*ptr, "Load");
                for path in ref_levels(&pointee) {
                    self.cs.add_load(
                        NodeKey::value(*result).extend(&path),
                        NodeKey::value(*ptr),
                        path,
                    );
                }
            }

            // --- Array / slice element flow (collapsed to AllElems for aliasing). ---
            OpCode::MkSeq { result, elems, .. } => {
                let elem_ty = self.array_element(*result);

                // Split: each literal position is its own `Index(i)` cell. Collapsed: all to
                // AllElems.
                let split = self.cells.split_indices(*result).is_some();
                for (i, elem) in elems.iter().enumerate() {
                    let cell = if split {
                        Cell::Index(i)
                    } else {
                        Cell::AllElems
                    };
                    self.copy_levels(
                        self.cell_node(*result, cell),
                        NodeKey::value(*elem),
                        &elem_ty,
                    );
                }
            }
            OpCode::MkRepeated {
                result, element, ..
            } => {
                // Always a collapse trigger — one source for every cell.
                let elem_ty = self.array_element(*result);
                self.copy_levels(self.elem_node(*result), NodeKey::value(*element), &elem_ty);
            }
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                let result_ty = self.value_type(*result).clone();
                if self.cells.split_indices(*array).is_some() {
                    // A Split group is only ever constant-indexed, so the index resolves.
                    let k = self
                        .const_index(*index)
                        .expect("ICE: Split array group read by a non-constant index");
                    self.copy_levels(
                        NodeKey::value(*result),
                        self.cell_node(*array, Cell::Index(k)),
                        &result_ty,
                    );
                } else {
                    self.copy_levels(NodeKey::value(*result), self.elem_node(*array), &result_ty);
                }
            }
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                let elem_ty = self.array_element(*result);
                let split: Option<Vec<usize>> = self
                    .cells
                    .split_indices(*result)
                    .map(|idx| idx.iter().copied().collect());
                match split {
                    Some(indices) => {
                        // `result` and `array` share the flow-group's index set. Overlay the
                        // written cell; inherit every other live cell (the wholesale-copy
                        // obligation).
                        let k = self
                            .const_index(*index)
                            .expect("ICE: Split array group set by a non-constant index");
                        for j in indices {
                            if j == k {
                                self.copy_levels(
                                    self.cell_node(*result, Cell::Index(k)),
                                    NodeKey::value(*value),
                                    &elem_ty,
                                );
                            } else {
                                self.copy_levels(
                                    self.cell_node(*result, Cell::Index(j)),
                                    self.cell_node(*array, Cell::Index(j)),
                                    &elem_ty,
                                );
                            }
                        }
                    }
                    None => {
                        self.copy_levels(self.elem_node(*result), self.elem_node(*array), &elem_ty);
                        self.copy_levels(self.elem_node(*result), NodeKey::value(*value), &elem_ty);
                    }
                }
            }
            OpCode::SlicePush {
                result,
                slice,
                values,
                ..
            } => {
                let elem_ty = self.array_element(*result);
                self.copy_levels(self.elem_node(*result), self.elem_node(*slice), &elem_ty);
                for value in values {
                    self.copy_levels(self.elem_node(*result), NodeKey::value(*value), &elem_ty);
                }
            }

            // --- Calls: instantiate the callee summary (replaces blanket escape). ---
            OpCode::Call {
                results,
                function: CallTarget::Static(callee),
                args,
                unconstrained,
            } => {
                self.instantiate_call(*callee, args, results, *unconstrained);
            }
            OpCode::Call {
                function: CallTarget::Dynamic(_),
                ..
            } => panic!("ICE: dynamic call target during points-to analysis"),

            // --- Globals. ---
            OpCode::ReadGlobal {
                result,
                offset,
                result_type,
            } => {
                // Each global slot+level is a distinct (escaped, opaque) `Global` object: two reads
                // of one slot alias, but a read does not alias unrelated locals — avoiding the
                // `External` virality that would block a coexisting local's split. Loads *through*
                // the read ref still yield `External` (Global is opaque), so the deref stays sound.
                let off = *offset as usize;
                for path in ref_levels(result_type) {
                    self.cs.add_base(
                        NodeKey::value(*result).extend(&path),
                        AbstractObject::Global(off, path),
                    );
                }
            }
            OpCode::InitGlobal { value, .. } => {
                let ty = self.value_type(*value).clone();
                self.escape_value(*value, &ty);
            }

            // --- Guard: delegate to the inner op (should not appear pre-WTI, but stay robust). ---
            OpCode::Guard { inner, .. } => self.build_instr(inner),

            // --- No pointer flow. ---
            OpCode::Cmp { .. }
            | OpCode::BinaryArithOp { .. }
            | OpCode::ToBits { .. }
            | OpCode::ToRadix { .. }
            | OpCode::SliceLen { .. }
            | OpCode::MkSeqOfBlob { .. }
            | OpCode::WriteWitness { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::Assert { .. }
            | OpCode::AssertCmp { .. }
            | OpCode::AssertR1C { .. }
            | OpCode::Rangecheck { .. }
            | OpCode::MemOp { .. } => {
                // Soundness invariant: a "no flow" opcode must not define a ref-bearing result.
                //
                // If one ever did, its result would get an empty points-to set and escape/alias
                // analysis would silently miss it, letting mem2reg / arg_promotion perform an
                // unsound strong update or promotion. This arm is hand-maintained, so pin the
                // invariant rather than trust it: a future ref-typed result trips here instead of
                // miscompiling. (Asserted via `ref_levels`, the same predicate the ref-aware
                // Store/Load/copy arms use to decide whether a value carries a pointer.)
                debug_assert!(
                    instr
                        .get_results()
                        .all(|r| ref_levels(self.value_type(*r)).is_empty()),
                    "ICE: points-to 'no pointer flow' opcode defines a ref-bearing result: {instr:?}",
                );
            }

            OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } | OpCode::MkTuple { .. } => {
                ice_non_elided_tuple()
            }

            OpCode::FreshWitness { .. }
            | OpCode::Constrain { .. }
            | OpCode::BumpD { .. }
            | OpCode::NextDCoeff { .. }
            | OpCode::MulConst { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. }
            | OpCode::Todo { .. } => {
                panic!("ICE: opcode should not be present during points-to analysis: {instr:?}")
            }
        }
    }

    /// Instantiate callee `g`'s summary at a call site: flow returned objects into the results
    /// (substituting `g`'s placeholders by the caller's actual argument objects), replay the
    /// callee's precise arg-out writes into caller memory, and escape any leaked arguments.
    ///
    /// Unconstrained calls and summary-less callees fall back to the sound `External`/escape model.
    fn instantiate_call(
        &mut self,
        g: FunctionId,
        args: &[ValueId],
        results: &[ValueId],
        unconstrained: bool,
    ) {
        // `self.summaries` is a shared reference, so copy it out: the chosen summary then borrows
        // for the builder's lifetime `'a` (its maps live outside `*self`), letting us read them
        // while the `&mut self` emission calls below mutate `self.cs`. This avoids deep-cloning the
        // summary's `returns` / `param_writes` / `leaks_param` maps at every call site, every
        // rebuild (a hot path: the summary fixpoint, the polymorphic pass, and per-context
        // specialization all re-run this).
        let summaries = self.summaries;
        let summary = if unconstrained {
            None
        } else {
            summaries.get(&g)
        };
        let Some(summary) = summary else {
            for arg in args {
                let ty = self.value_type(*arg).clone();
                self.escape_value(*arg, &ty);
            }
            for result in results {
                let ty = self.value_type(*result).clone();
                self.seed_external(*result, &ty);
            }
            return;
        };

        // Defensive: every `results[j]` / `args[i]` index below comes from the callee summary's
        // return-slot / parameter indices, and relies on the IR invariant that a static `Call`'s
        // result/argument arity matches its callee's signature exactly. That holds by construction
        // (Unit-driven return count; 1:1 params from the same monomorphized AST), so this can only
        // trip on a malformed call — turning a raw out-of-bounds index panic into a clear ICE.
        debug_assert!(
            summary.returns.keys().all(|(j, _)| *j < results.len())
                && summary.param_writes.keys().all(|(i, _, _)| *i < args.len())
                && summary.leaks_param.iter().all(|i| *i < args.len()),
            "ICE: callee {g:?} summary indexes beyond the call site arity \
             (results={}, args={})",
            results.len(),
            args.len(),
        );

        let callee_ctx = self.callee_ctx(args, results);

        // Record the reachable callee context for the Phase-2 BFS.
        self.callee_contexts.push((g, callee_ctx.clone()));

        for ((j, path), objset) in &summary.returns {
            let target = NodeKey::Val(Owner::Value(results[*j]), path.clone());
            for o in objset {
                self.emit_object_into(&target, o, args, &callee_ctx);
            }
        }

        // Precise arg-out: store the exact objects the callee wrote into the caller's arg memory.
        for ((i, plpath, cellpath), objset) in &summary.param_writes {
            let ptr = NodeKey::Val(Owner::Value(args[*i]), plpath.clone());
            for o in objset {
                self.store_summand(&ptr, cellpath, o, args, &callee_ctx);
            }
        }
        for i in &summary.leaks_param {
            let arg_ty = self.value_type(args[*i]).clone();
            self.escape_value(args[*i], &arg_ty);
        }
    }

    /// Store one arg-out summand `o` into the caller's arg-pointee cell `Obj(*, cellpath)` reached
    /// through `ptr`, applying the same substitution as `emit_object_into`.
    ///
    /// A placeholder stores straight from the caller's matching argument level; a callee local is
    /// re-contextualized; a concrete sink stores from its canonical `obj_ref` node.
    fn store_summand(
        &mut self,
        ptr: &NodeKey,
        cellpath: &Path,
        o: &AbstractObject,
        args: &[ValueId],
        callee_ctx: &Context,
    ) {
        let src = match o {
            AbstractObject::Placeholder(_, j, q) => NodeKey::Val(Owner::Value(args[*j]), q.clone()),
            AbstractObject::Alloc(f2, v, _) => {
                self.obj_ref(AbstractObject::Alloc(*f2, *v, callee_ctx.clone()))
            }
            AbstractObject::Global(..) | AbstractObject::External => self.obj_ref(o.clone()),
        };
        self.cs.add_store(ptr.clone(), cellpath.clone(), src);
    }

    /// Emit a single summary object `o` into caller node `target`, applying the substitution.
    ///
    /// A callee placeholder becomes a copy from the matching caller argument level; a callee local
    /// allocation is re-qualified into the call's context; concrete sinks pass through.
    fn emit_object_into(
        &mut self,
        target: &NodeKey,
        o: &AbstractObject,
        args: &[ValueId],
        callee_ctx: &Context,
    ) {
        match o {
            AbstractObject::Placeholder(_, i, p) => {
                self.cs.add_copy(
                    target.clone(),
                    NodeKey::Val(Owner::Value(args[*i]), p.clone()),
                );
            }
            AbstractObject::Alloc(f2, v, _) => {
                self.cs.add_base(
                    target.clone(),
                    AbstractObject::Alloc(*f2, *v, callee_ctx.clone()),
                );
            }
            AbstractObject::Global(..) | AbstractObject::External => {
                self.cs.add_base(target.clone(), o.clone());
            }
        }
    }

    /// The element-level node of an array/slice value (its collapsed `AllElems` cell).
    fn elem_node(&self, array: ValueId) -> NodeKey {
        NodeKey::value(array).extend(&[Descent::Elem(Cell::AllElems)])
    }

    /// The element type of an array/slice-typed value.
    fn array_element(&self, v: ValueId) -> Type {
        self.value_type(v).peel_witness().get_array_element()
    }

    /// The pointee type of a ref-typed value, or ICE with `op` context.
    fn pointee_of(&self, ptr: ValueId, op: &str) -> Type {
        match &self.value_type(ptr).peel_witness().expr {
            TypeExpr::Ref(inner) => (**inner).clone(),
            other => panic!("ICE: {op} through a non-ref value of type {other:?}"),
        }
    }
}

// TYPE-LEVEL HELPERS
// ================================================================================================

/// Every path to a `Ref`-typed level of `ty`, descending only through array/slice elements (which
/// collapse to [`Cell::AllElems`]).
///
/// These are the levels at which a *value* of type `ty` directly holds a pointer; a `Ref`'s own
/// pointee is object-land, not a value level, so descent stops at the first `Ref`.
///
/// This is deliberately *not* `witness_taint_inference`'s `paths_of_type`/`leaf_paths`: the former
/// descends through `Deref` into pointees (which points-to must not do — a pointee is reached
/// through the object graph, never a value's own path), and the latter also emits scalar leaves and
/// lacks the [`Cell`] payload. Deriving ref-only paths from either is more code than this focused
/// walker, and a small per-analysis type walker is the house idiom.
pub fn ref_levels(ty: &Type) -> Vec<Path> {
    let mut out = Vec::new();
    let mut prefix = Vec::new();
    collect_ref_levels(ty, &mut prefix, &mut out);
    out
}

fn collect_ref_levels(ty: &Type, prefix: &mut Path, out: &mut Vec<Path>) {
    let ty = ty.peel_witness();
    match &ty.expr {
        TypeExpr::Ref(_) => out.push(prefix.clone()),
        TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
            prefix.push(Descent::Elem(Cell::AllElems));
            collect_ref_levels(inner, prefix, out);
            prefix.pop();
        }
        TypeExpr::Field
        | TypeExpr::U(_)
        | TypeExpr::I(_)
        | TypeExpr::Function
        | TypeExpr::Blob(..) => {}
        TypeExpr::WitnessOf(_) => unreachable!("peeled above"),
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
    }
}
