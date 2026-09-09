//! Purifies witness-length slices into `(physical, log_len, start)` tuples.
//!
//! A witness-length slice becomes a pure-length slice observed through the window
//! `physical[start .. start + log_len]`; outside slots are garbage, and every rewrite maintains
//! `start + log_len <= physical.len()`.
//!
//! Which values become windows is decided from the read-only [`ApproximateWitnessTaint`], whose
//! shapes are per function. A representation is not: [`close_boundaries`] closes the decision over
//! every call, return and merge before a single instruction is rewritten.
//!
//! Cost note: a witnessed push appends to `physical`, then `array_set`s the value. The
//! witness-indexed set lowers to a rebuild loop, so n pushes cost O(n^2) constraints.
//!
//! Must run before `ElideTuples` and `UntaintControlFlow`. Reads the read-only
//! [`ApproximateWitnessTaint`].

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            types::{FunctionTypeInfo, TypeInfo},
            witness_info::WitnessShape,
            witness_taint_inference::ApproximateWitnessTaint,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        ssa::{
            BlockId, FunctionId, Instruction, SourceLocation, Terminator, ValueId,
            hlssa::{
                CallTarget, CastTarget, CmpKind, HLFunction, HLSSA, LocatedOpCode, OpCode,
                SliceOpDir, Type, TypeExpr,
                builder::{HLEmitter, HLInstrBuilder},
            },
        },
    },
};

pub struct PurifyWitnessSlices {}

impl PurifyWitnessSlices {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for PurifyWitnessSlices {
    fn name(&self) -> &'static str {
        "purify_witness_slices"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let flow = store.get::<FlowAnalysis>();
        let types = store.get::<TypeInfo>();
        let approx = ApproximateWitnessTaint::compute(ssa, flow, types);

        let function_ids: Vec<_> = ssa.get_function_ids().collect();

        let mut lifts = BoundaryLifts::default();
        let mut affected: HashMap<FunctionId, HashMap<ValueId, Type>> = HashMap::default();
        for &function_id in &function_ids {
            if !types.has_function(function_id) {
                continue;
            }
            let Some(value_shapes) = approx.value_shapes(function_id) else {
                continue;
            };
            let function = ssa.get_function(function_id);
            let params = function
                .get_entry()
                .get_parameters()
                .map(|(v, ty)| {
                    value_shapes
                        .get(v)
                        .is_some_and(|shape| is_slice_tuple(&purify_type(ty, shape)))
                })
                .collect();
            let return_shapes = approx.return_shapes(function_id).unwrap();

            debug_assert_eq!(
                function.get_returns().len(),
                return_shapes.len(),
                "purify_witness_slices: {function_id:?} has {} return slots but {} shapes; the zip below would silently drop the excess",
                function.get_returns().len(),
                return_shapes.len()
            );

            let returns = function
                .get_returns()
                .iter()
                .zip(return_shapes)
                .map(|(ty, shape)| is_slice_tuple(&purify_type(ty, shape)))
                .collect();
            lifts.params.insert(function_id, params);
            lifts.returns.insert(function_id, returns);
            affected.insert(
                function_id,
                affected_values(types.get_function(function_id), value_shapes),
            );
        }

        close_boundaries(ssa, types, &mut affected, &mut lifts);

        for function_id in function_ids {
            let Some(affected) = affected.remove(&function_id) else {
                continue;
            };
            let type_info = types.get_function(function_id);
            let returns_witness = approx.return_shapes(function_id).unwrap().to_vec();

            let block_order: Vec<BlockId> = flow
                .get_function_cfg(function_id)
                .get_domination_pre_order()
                .collect();
            let mut function = ssa.take_function(function_id);
            rewrite_function(
                &mut function,
                ssa,
                type_info,
                &affected,
                &block_order,
                &returns_witness,
                &lifts.returns[&function_id],
                &lifts,
            );
            ssa.put_function(function_id, function);
        }

        for function_id in ssa.get_function_ids().collect::<Vec<_>>() {
            if types.has_function(function_id) && approx.value_shapes(function_id).is_some() {
                continue;
            }
            let no_lifted_slots = |m: &HashMap<FunctionId, Vec<bool>>, callee: FunctionId| {
                !m.get(&callee).is_some_and(|l| l.iter().any(|&x| x))
            };
            let function = ssa.get_function(function_id);
            for (_, block) in function.get_blocks() {
                for instr in block.get_instructions() {
                    for callee in instr.get_static_call_targets() {
                        assert!(
                            no_lifted_slots(&lifts.params, callee)
                                && no_lifted_slots(&lifts.returns, callee),
                            "ICE: purify_witness_slices: skipped function {function_id:?} calls {callee:?}, whose purified signature exposes slice tuples"
                        );
                    }
                }
            }
        }
    }
}

/// Which by-value param/return slots of each function carry a window.
///
/// Seeded from each function's own joined shapes and then closed over the whole program by
/// [`close_boundaries`] — see its doc for why the seed alone is not a fixed point.
#[derive(Default)]
struct BoundaryLifts {
    params: HashMap<FunctionId, Vec<bool>>,
    returns: HashMap<FunctionId, Vec<bool>>,
}

/// The `(physical, log_len, start)` window over a physical slice type.
fn window_type(physical: Type) -> Type {
    Type::tuple_of(vec![physical, Type::int(32), Type::int(32)])
}

fn purify_type(ty: &Type, shape: &WitnessShape) -> Type {
    match (&ty.expr, shape) {
        (TypeExpr::Slice(elem), WitnessShape::Slice(len_type, inner)) => {
            let physical = ice_on_nested_window(purify_type(elem, inner)).slice_of();
            if len_type.is_witness() {
                window_type(physical)
            } else {
                physical
            }
        }
        (TypeExpr::Ref(inner_ty), WitnessShape::Ref(_, inner_shape)) => {
            purify_type(inner_ty, inner_shape).ref_of()
        }
        (TypeExpr::Array(elem, n), WitnessShape::Array(inner)) => {
            ice_on_nested_window(purify_type(elem, inner)).array_of(*n)
        }
        _ => ty.clone(),
    }
}

/// Refuse an array/slice *element* that purified into a window.
///
/// This is the `[[Field]]` shape whose inner vector is witness-length. No arm of this pass
/// rewrites it, and unlike every other unsupported shape it does not fail loudly on its own: an
/// *instruction* producing or consuming one ICEs in the `other` fallback, but the block-parameter
/// path in [`rewrite_function`] retypes the parameter from `affected` while its jump arguments
/// pass through untouched (`lifted_block_args` only marks the window parameters, and the `Jmp`
/// arm's assert only fires for values already in `replacement_tuple_map`), leaving ill-typed SSA
/// behind with no diagnostic at all.
///
/// Refusing where the shape is *minted* covers every consumer at once — parameters, returns, and
/// [`affected_values`] all reach it through [`purify_type`]. A window behind a `Ref` is fine and
/// deliberately not checked: that is the ordinary `&mut [Field]` case, and a ref's pointee type is
/// inferred from its `Alloc`, so it tracks the rewrite on its own.
fn ice_on_nested_window(purified_elem: Type) -> Type {
    assert!(
        !is_slice_tuple(&purified_elem),
        "ICE: purify_witness_slices: witness-length slice nested in a container; this pass has no \
         rewrite for a window nested in a container element"
    );
    purified_elem
}

fn affected_values(
    type_info: &FunctionTypeInfo,
    value_shapes: &HashMap<ValueId, WitnessShape>,
) -> HashMap<ValueId, Type> {
    let mut affected: HashMap<ValueId, Type> = HashMap::default();
    for (&v, shape) in value_shapes.iter() {
        let ty = type_info.get_value_type(v);
        let pty = purify_type(ty, shape);
        if pty != *ty {
            affected.insert(v, pty);
        }
    }
    affected
}

// BOUNDARY CLOSURE
// ================================================================================================

/// Values whose purified type [`close_boundaries`] may widen into a window.
///
/// A slice value is self-contained: the rewrite reads its type straight out of `affected`, so
/// widening one changes nothing it cannot see. A **ref** is not — widening a ref retypes the *cell*
/// it names, and every other alias of that cell has to be retyped with it. This pass reaches only
/// the aliases it can see structurally, so `cells` accepts a cell's own `Alloc` result (whose
/// initializer arm reads the very same entry) or a parameter (whose callers the fixed point widens
/// in turn), and refuses the rest rather than retype half of an aliased cell.
struct Widenable {
    values: HashSet<ValueId>,
    cells: HashSet<ValueId>,
}

fn widenable_values(function: &HLFunction) -> Widenable {
    let mut values: HashSet<ValueId> = HashSet::default();
    let mut cells: HashSet<ValueId> = HashSet::default();
    for (_, block) in function.get_blocks() {
        for (v, _) in block.get_parameters() {
            values.insert(*v);
            cells.insert(*v);
        }
        for instr in block.get_instructions() {
            match instr {
                OpCode::Alloc { result, .. } => {
                    values.insert(*result);
                    cells.insert(*result);
                }
                OpCode::SlicePush { result, .. }
                | OpCode::SliceInsert { result, .. }
                | OpCode::ArraySet { result, .. }
                | OpCode::Cast { result, .. }
                | OpCode::Select { result, .. }
                | OpCode::Load { result, .. } => {
                    values.insert(*result);
                }
                OpCode::SlicePop { result_slice, .. }
                | OpCode::SliceRemove { result_slice, .. } => {
                    values.insert(*result_slice);
                }
                OpCode::Call { results, .. } => values.extend(results.iter().copied()),
                _ => {}
            }
        }
    }
    Widenable { values, cells }
}

/// The values of `function` that will carry a window, given what its own shapes say (`affected`)
/// and which callee slots are lifted (`lifts`).
///
/// Mirrors — and only mirrors — the *operand*-driven half of [`rewrite_instruction`]'s
/// window-producing conditions; the result-driven half is already in `affected`, which seeds this.
/// [`rewrite_function`] debug-asserts its `replacement_tuple_map` against the closed `affected`, so
/// a rule added there without a rule here fails loudly rather than silently under-approximating.
fn plan_windows(
    function: &HLFunction,
    affected: &HashMap<ValueId, Type>,
    lifts: &BoundaryLifts,
) -> HashSet<ValueId> {
    let mut windows: HashSet<ValueId> = affected
        .iter()
        .filter(|(_, ty)| is_slice_tuple(ty))
        .map(|(v, _)| *v)
        .collect();

    loop {
        let mut changed = false;
        for (_, block) in function.get_blocks() {
            for instr in block.get_instructions() {
                match instr {
                    OpCode::SlicePush { result, slice, .. }
                    | OpCode::SliceInsert { result, slice, .. } => {
                        changed |= windows.contains(slice) && windows.insert(*result);
                    }
                    OpCode::SlicePop {
                        result_slice,
                        slice,
                        ..
                    }
                    | OpCode::SliceRemove {
                        result_slice,
                        slice,
                        ..
                    } => {
                        changed |= windows.contains(slice) && windows.insert(*result_slice);
                    }
                    OpCode::ArraySet { result, array, .. } => {
                        changed |= windows.contains(array) && windows.insert(*result);
                    }
                    OpCode::Cast { result, value, .. } => {
                        changed |= windows.contains(value) && windows.insert(*result);
                    }

                    // A window in either alternative makes the result one. Widening the result is
                    // the only direction that reconciles the two arms: the rewrite's `arm` closure
                    // can materialize a *pure* alternative into a window, but nothing can narrow a
                    // window back into a bare slice.
                    OpCode::Select {
                        result, if_t, if_f, ..
                    } => {
                        changed |= (windows.contains(if_t) || windows.contains(if_f))
                            && windows.insert(*result);
                    }
                    OpCode::Load { result, ptr } => {
                        changed |= is_ref_to_wl_slice(*ptr, affected) && windows.insert(*result);
                    }
                    OpCode::Call {
                        results,
                        function: CallTarget::Static(callee),
                        ..
                    } => {
                        let lifted = lifts.returns.get(callee).map(Vec::as_slice).unwrap_or(&[]);
                        for (i, result) in results.iter().enumerate() {
                            if lifted.get(i).copied().unwrap_or(false) {
                                changed |= windows.insert(*result);
                            }
                        }
                    }
                    _ => {}
                }
            }
            // A window jumped into a merge makes that merge parameter one too.
            if let Some(Terminator::Jmp(target, args)) = block.get_terminator() {
                let params: Vec<ValueId> = function
                    .get_block(*target)
                    .get_parameters()
                    .map(|(v, _)| *v)
                    .collect();
                for (i, arg) in args.iter().enumerate() {
                    if let Some(&param) = params.get(i) {
                        changed |= windows.contains(arg) && windows.insert(param);
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    windows
}

fn widen(
    changed: &mut bool,
    affected: &mut HashMap<FunctionId, HashMap<ValueId, Type>>,
    widenable: &HashMap<FunctionId, Widenable>,
    fid: FunctionId,
    v: ValueId,
    ty: Type,
) {
    let slot = affected.get_mut(&fid).expect("a purified function");
    if slot.get(&v) == Some(&ty) {
        return;
    }
    let allowed = if ty.is_ref() {
        &widenable[&fid].cells
    } else {
        &widenable[&fid].values
    };
    assert!(
        allowed.contains(&v),
        "ICE: purify_witness_slices: v{} in {fid:?} has to become a slice window to agree with a \
         call boundary, but this pass cannot reach every alias of it",
        v.0
    );
    slot.insert(v, ty);
    *changed = true;
}

/// Close "is a window" over every boundary a window can cross.
///
/// Everything above this point is decided per function, from that function's *own* joined shapes.
/// A value's representation is not a per-function property, though: a callee is purified at the
/// join over all of its call sites, so it can decide a slot is a window while a caller whose own
/// site still calls it pure-length holds a bare slice.
///
/// [`is_window_op`] and the `Call` arm already absorb that skew *forward* — a lifted callee's
/// result is a window at every site, and the ops consuming it follow. What they cannot do is carry
/// it back out: once such a window exists it can reach a return slot, another callee's parameter,
/// a merge parameter, or one alternative of a `Select` that this function's shapes still call pure,
/// and each of those is a hard refusal ("wl slice tuple flows into a pure").
///
/// A `Select` differs only in which side gives way — its two alternatives have to agree with each
/// other, so it is the *result* that widens rather than the slot the window flows into. A `&mut`
/// cell is worse — the callee writes through the pointer, so there is nothing to materialize at the
/// boundary at all. A copy-out shim would have to turn a window back into a witness-length slice,
/// which is precisely what has no representation as the caller's own cell would have to become a
/// window instead or the two sides disagree on the pointee type and the mismatch surfaces far away
/// as an error in `Types`.
///
/// As a result, we run the seed to a fixed point. Widening is always safe: a pure-length slice held
/// as a window is `(physical, physical.len(), 0)`, whose reads fold straight back to the constants
/// they came from. It terminates because both maps only ever grow inside a finite domain with one
/// bit per boundary slot, one entry per value.
fn close_boundaries(
    ssa: &HLSSA,
    types: &TypeInfo,
    affected: &mut HashMap<FunctionId, HashMap<ValueId, Type>>,
    lifts: &mut BoundaryLifts,
) {
    let fids: Vec<FunctionId> = affected.keys().copied().collect();
    let widenable: HashMap<FunctionId, Widenable> = fids
        .iter()
        .map(|&fid| (fid, widenable_values(ssa.get_function(fid))))
        .collect();

    loop {
        let mut changed = false;
        for &fid in &fids {
            let function = ssa.get_function(fid);
            let windows = plan_windows(function, &affected[&fid], lifts);

            // Every planned window has to *say* it is one, or the rewrite would retype neither the
            // merge parameter nor the boundary slot carrying it.
            for &v in &windows {
                if is_wl_slice(v, &affected[&fid]) {
                    continue;
                }
                let physical = physical_type(v, &affected[&fid], types.get_function(fid));
                widen(
                    &mut changed,
                    affected,
                    &widenable,
                    fid,
                    v,
                    window_type(physical),
                );
            }

            for (_, block) in function.get_blocks() {
                // A window reaching a return slot lifts it for every caller.
                if let Some(Terminator::Return(values)) = block.get_terminator() {
                    for (i, v) in values.iter().enumerate() {
                        let slots = lifts.returns.get_mut(&fid).expect("a purified function");
                        // `.get`, not an index: the arity is debug-asserted in `rewrite_function`,
                        // and this must not be the place a mismatch first turns into a panic.
                        if windows.contains(v) && slots.get(i) == Some(&false) {
                            slots[i] = true;
                            changed = true;
                        }
                    }
                }

                for instr in block.get_instructions() {
                    // A window put into a cell makes that cell a window — the reverse of the
                    // `Alloc`/`Store` arms' materialization, which only bridges the pure value
                    // into an already-windowed slot. The cell can acquire one without this
                    // function's shapes ever saying so: a lifted callee hands back a window that
                    // is then stored.
                    match instr {
                        OpCode::Alloc { result, value } if windows.contains(value) => {
                            let ty = affected[&fid][value].clone().ref_of();
                            widen(&mut changed, affected, &widenable, fid, *result, ty);
                        }
                        OpCode::Store { ptr, value } if windows.contains(value) => {
                            let ty = affected[&fid][value].clone().ref_of();
                            widen(&mut changed, affected, &widenable, fid, *ptr, ty);
                        }
                        _ => {}
                    }

                    let OpCode::Call {
                        results,
                        function: CallTarget::Static(callee),
                        args,
                        ..
                    } = instr
                    else {
                        continue;
                    };
                    // A callee this pass skipped has no lifted slot to reconcile: `run`'s closing
                    // assert refuses the reverse (a skipped function calling a lifted one).
                    if !affected.contains_key(callee) {
                        continue;
                    }
                    let formals: Vec<ValueId> = ssa
                        .get_function(*callee)
                        .get_entry()
                        .get_parameters()
                        .map(|(v, _)| *v)
                        .collect();

                    for (i, arg) in args.iter().enumerate() {
                        let Some(&formal) = formals.get(i) else {
                            continue;
                        };

                        // By value: a window argument lifts the slot, and a lifted slot makes the
                        // callee's formal a window. The *caller* side needs no widening — a pure
                        // argument is bridged by `materialize_pure_slice_tuple`.
                        if windows.contains(arg) && !lifts.params[callee][i] {
                            lifts.params.get_mut(callee).expect("a purified function")[i] = true;
                            changed = true;
                        }
                        if lifts.params[callee][i] && !is_wl_slice(formal, &affected[callee]) {
                            let physical = physical_type(
                                formal,
                                &affected[callee],
                                types.get_function(*callee),
                            );
                            widen(
                                &mut changed,
                                affected,
                                &widenable,
                                *callee,
                                formal,
                                window_type(physical),
                            );
                        }

                        // Through a `&mut`: the two sides name one cell, so they must agree on it.
                        let caller_cell = is_ref_to_wl_slice(*arg, &affected[&fid]);
                        let callee_cell = is_ref_to_wl_slice(formal, &affected[callee]);
                        match (caller_cell, callee_cell) {
                            (true, false) => {
                                let ty = affected[&fid][arg].clone();
                                widen(&mut changed, affected, &widenable, *callee, formal, ty);
                            }
                            (false, true) => {
                                let ty = affected[callee][&formal].clone();
                                widen(&mut changed, affected, &widenable, fid, *arg, ty);
                            }
                            _ => {}
                        }
                    }

                    for (i, result) in results.iter().enumerate() {
                        if lifts.returns[callee].get(i).copied().unwrap_or(false)
                            && !is_wl_slice(*result, &affected[&fid])
                        {
                            let physical =
                                physical_type(*result, &affected[&fid], types.get_function(fid));
                            widen(
                                &mut changed,
                                affected,
                                &widenable,
                                fid,
                                *result,
                                window_type(physical),
                            );
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Whether `ty` is this pass's `(physical, log_len, start)` window representation.
///
/// Matching the exact shape rather than "is a tuple" keeps the predicate safe if [`purify_type`]
/// ever grows a `TypeExpr::Tuple` arm (say to reach a witness-length slice nested in a struct).
/// Today no other type is rewritten *into* a tuple, so "is a tuple" happens to be equivalent — but
/// the day that changes, a user struct would start reading as a slice window and [`physical_type`]
/// would hand its first field to the slice rewrites.
fn is_slice_tuple(ty: &Type) -> bool {
    let TypeExpr::Tuple(elems) = &ty.expr else {
        return false;
    };
    matches!(
        elems.as_slice(),
        [physical, log_len, start]
            if physical.is_slice() && *log_len == Type::int(32) && *start == Type::int(32)
    )
}

fn is_wl_slice(v: ValueId, affected: &HashMap<ValueId, Type>) -> bool {
    affected.get(&v).is_some_and(is_slice_tuple)
}

/// Whether a slice op must be rewritten into its windowed form.
///
/// There are two independent reasons, and tripping _either_ is sufficient:
///
/// - The result is a witness-length slice by this function's own shapes (`affected`). This is the
///   common case: the op is what makes the length witness-dependent.
/// - The *operand* already carries a window (`replacement_tuple_map`), whatever the shapes say
///   about the result. This is not redundant. `affected` holds per-call-site shapes, while the
///   purified call signatures in [`BoundaryLifts`] come from the callee's joined summary, but the
///   join over *every* site. So a callee whose return is witness-length at some other site returns
///   a window here too, even where this site's shapes call the result pure. Keying only on the
///   result would then leave the op untouched and hand a `(physical, log_len, start)` tuple to a
///   rewrite expecting a bare slice, which would fail much later in `Types` with a bewildering
///   error.
///
/// Because the rewrite registers its own result in `replacement_tuple_map`, this second reason
/// propagates down a chain of slice ops on its own — the map *is* the forward closure.
///
/// Forward is all it is, though: a window this reason creates can go on to reach a boundary — a
/// return slot, another callee's parameter, a merge parameter — that this function's shapes still
/// call pure, and every one of those is a refusal rather than a rewrite. [`close_boundaries`]
/// raises those slots up front so they never come up; this predicate is what tells it where the
/// windows are.
fn is_window_op(
    result: ValueId,
    operand: ValueId,
    affected: &HashMap<ValueId, Type>,
    replacement_tuple_map: &HashMap<ValueId, ValueId>,
) -> bool {
    is_wl_slice(result, affected) || replacement_tuple_map.contains_key(&operand)
}

fn debug_assert_u32_index(index: ValueId, type_info: &FunctionTypeInfo) {
    let ty = type_info.get_value_type(index);
    debug_assert!(
        matches!(ty.strip_witness().expr, TypeExpr::Int(32)),
        "purify_witness_slices: slice index must be u32, got {ty}"
    );
}

fn is_ref_to_wl_slice(v: ValueId, affected: &HashMap<ValueId, Type>) -> bool {
    affected
        .get(&v)
        .is_some_and(|pty| pty.is_ref() && is_slice_tuple(pty.get_refered()))
}

/// The physical slice type behind `v` — the `[0]` component if `v` is a window, else `v`'s own
/// (elementwise-purified) slice type.
///
/// The second case is not a fallback for missing information: a value can be *window-valued*
/// without being in `affected`, because `affected` holds this call site's shapes while the purified
/// call signatures in [`BoundaryLifts`] come from the callee's joined summary, which is the join
/// over every site (see the `Call` arm). A slice op on such a value is still a window op — and the
/// physical type is the same either way, since purification only ever moves the *length* out, never
/// changes the element type.
fn physical_type(
    v: ValueId,
    affected: &HashMap<ValueId, Type>,
    type_info: &FunctionTypeInfo,
) -> Type {
    match affected.get(&v) {
        Some(pty) if is_slice_tuple(pty) => pty.get_tuple_elements()[0].clone(),
        Some(pty) => pty.clone(),
        None => type_info.get_value_type(v).clone(),
    }
}

fn mk_slice_tuple(
    physical: ValueId,
    log_len: ValueId,
    start: ValueId,
    phys_ty: Type,
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    new_instrs: &mut Vec<LocatedOpCode>,
    loc: SourceLocation,
) -> ValueId {
    let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc);
    b.mk_tuple(
        vec![physical, log_len, start],
        vec![phys_ty, Type::int(32), Type::int(32)],
    )
}

fn materialize_pure_slice_tuple(
    slice: ValueId,
    type_info: &FunctionTypeInfo,
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    new_instrs: &mut Vec<LocatedOpCode>,
) -> ValueId {
    let phys_ty = type_info.get_value_type(slice).clone();
    let loc = SourceLocation::synthetic("purify_witness_slices");
    let (ll, start) = {
        let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone());
        (b.slice_len(slice), b.int_const(32, 0))
    };
    mk_slice_tuple(slice, ll, start, phys_ty, function, ssa, new_instrs, loc)
}

fn rewrite_function(
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    type_info: &FunctionTypeInfo,
    affected: &HashMap<ValueId, Type>,
    block_order: &[BlockId],
    returns_witness: &[WitnessShape],
    lifted_returns: &[bool],
    lifts: &BoundaryLifts,
) {
    let mut replacement_tuple_map: HashMap<ValueId, ValueId> = HashMap::default();

    let lifted_block_args: HashMap<BlockId, Vec<bool>> = function
        .get_blocks()
        .map(|(bid, block)| {
            let positions = block
                .get_parameters()
                .map(|(v, _)| is_wl_slice(*v, affected))
                .collect();
            (*bid, positions)
        })
        .collect();

    debug_assert_eq!(
        function.get_returns().len(),
        returns_witness.len(),
        "purify_witness_slices: {} return slots but {} shapes; the zip below would silently leave the excess slots un-purified",
        function.get_returns().len(),
        returns_witness.len()
    );

    // `lifted_returns` comes from the closed [`BoundaryLifts`], not from this function's own
    // shapes: a slot the closure raised carries a window here even though `purify_type` alone
    // would leave it a bare slice (see [`close_boundaries`]).
    for ((ty, shape), lifted) in function
        .iter_returns_mut()
        .zip(returns_witness)
        .zip(lifted_returns)
    {
        let pty = purify_type(ty, shape);
        *ty = if *lifted && !is_slice_tuple(&pty) {
            window_type(pty)
        } else {
            pty
        };
    }

    for &bid in block_order {
        let old_params = function.get_block_mut(bid).take_parameters();
        let mut new_params = Vec::with_capacity(old_params.len());
        for (v, ty) in old_params {
            if let Some(pty) = affected.get(&v) {
                new_params.push((v, pty.clone()));
                if is_wl_slice(v, affected) {
                    replacement_tuple_map.insert(v, v); // the param value IS the tuple
                }
            } else {
                new_params.push((v, ty));
            }
        }
        function.get_block_mut(bid).put_parameters(new_params);

        let old_instrs = function.get_block_mut(bid).take_instructions();
        let mut new_instrs: Vec<LocatedOpCode> = Vec::new();
        for located in old_instrs {
            let (op, loc) = located.take();
            rewrite_instruction(
                op,
                loc,
                function,
                ssa,
                type_info,
                affected,
                lifts,
                &mut replacement_tuple_map,
                &mut new_instrs,
            );
        }

        let terminator = function
            .get_block_mut(bid)
            .take_terminator()
            .expect("terminated block");
        let new_terminator = match terminator {
            Terminator::Jmp(target, args) => {
                let positions = &lifted_block_args[&target];
                let mut new_args = Vec::with_capacity(args.len());
                for (i, arg) in args.into_iter().enumerate() {
                    if positions.get(i).copied().unwrap_or(false) {
                        let t = replacement_tuple_map.get(&arg).copied().unwrap_or_else(|| {
                            // Materializing reads `arg`'s *pre-rewrite* type, so it is only valid
                            // for a value this pass left alone. Unreachable today (only a `Ref`
                            // purifies into something other than a window, and a ref cannot fill a
                            // slice-typed slot), but the `Alloc`/`Store`/`Select` arms all pin the
                            // same precondition and this one must not be the odd one out.
                            assert!(
                                !affected.contains_key(&arg),
                                "ICE: purify_witness_slices: jump argument v{} is affected but has no slice tuple",
                                arg.0
                            );
                            materialize_pure_slice_tuple(
                                arg,
                                type_info,
                                function,
                                ssa,
                                &mut new_instrs,
                            )
                        });
                        new_args.push(t);
                    } else {
                        assert!(
                            !replacement_tuple_map.contains_key(&arg),
                            "ICE: purify_witness_slices: wl slice tuple flows into a pure"
                        );
                        new_args.push(arg);
                    }
                }
                Terminator::Jmp(target, new_args)
            }
            Terminator::JmpIf(cond, t, f) => Terminator::JmpIf(cond, t, f),
            Terminator::Return(values) => {
                let mut new_return_args = Vec::with_capacity(values.len());
                for (i, v) in values.into_iter().enumerate() {
                    if lifted_returns.get(i).copied().unwrap_or(false) {
                        let t = replacement_tuple_map.get(&v).copied().unwrap_or_else(|| {
                            assert!(
                                !affected.contains_key(&v),
                                "ICE: purify_witness_slices: returned value v{} is affected but has no slice tuple",
                                v.0
                            );
                            materialize_pure_slice_tuple(
                                v,
                                type_info,
                                function,
                                ssa,
                                &mut new_instrs,
                            )
                        });
                        new_return_args.push(t);
                    } else {
                        assert!(
                            !replacement_tuple_map.contains_key(&v),
                            "ICE: purify_witness_slices: wl slice tuple flows into a pure"
                        );
                        new_return_args.push(v);
                    }
                }
                Terminator::Return(new_return_args)
            }
        };

        function.get_block_mut(bid).put_instructions(new_instrs);
        function.get_block_mut(bid).set_terminator(new_terminator);
    }

    // The drift guard between the rewrite and [`plan_windows`]: every window the rewrite actually
    // discovered must be one the closure predicted, or a boundary slot carrying it was left
    // un-lifted. A window-producing arm added above without the matching rule in `plan_windows`
    // fails here instead of much later, in whichever boundary it happens to escape through.
    debug_assert!(
        replacement_tuple_map
            .keys()
            .all(|v| is_wl_slice(*v, affected)),
        "purify_witness_slices: the rewrite produced a window `plan_windows` did not predict; the \
         two must agree or `close_boundaries` under-approximates"
    );
}

/// Rewrites one instruction into `new_instrs`, recording any slice-tuple replacement it
/// introduces.
fn rewrite_instruction(
    op: OpCode,
    loc: SourceLocation,
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    type_info: &FunctionTypeInfo,
    affected: &HashMap<ValueId, Type>,
    lifts: &BoundaryLifts,
    replacement_tuple_map: &mut HashMap<ValueId, ValueId>,
    new_instrs: &mut Vec<LocatedOpCode>,
) {
    match op {
        OpCode::SliceLen { result, slice } if replacement_tuple_map.contains_key(&slice) => {
            let t = replacement_tuple_map[&slice];
            new_instrs.push(
                OpCode::TupleProj {
                    result,
                    tuple: t,
                    idx: 1,
                }
                .locate(loc),
            );
        }

        OpCode::SlicePush {
            result,
            slice,
            values,
            dir,
        } if is_window_op(result, slice, affected, replacement_tuple_map) => {
            let phys_ty = physical_type(result, affected, type_info);
            let (physical, log_len, start) = {
                let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone());
                let bump = b.int_const(32, values.len() as u128);
                match (replacement_tuple_map.get(&slice).copied(), dir) {
                    (Some(t), SliceOpDir::Back) => {
                        let p = b.tuple_proj(t, 0);
                        let ll = b.tuple_proj(t, 1);
                        let st = b.tuple_proj(t, 2);
                        let one = b.int_const(32, 1);
                        let mut physical = p;
                        let mut cursor = b.uadd(st, ll);
                        for value in &values {
                            let grown = b.slice_push(physical, vec![*value], SliceOpDir::Back);
                            physical = b.array_set(grown, cursor, *value);
                            cursor = b.uadd(cursor, one);
                        }
                        (physical, b.uadd(ll, bump), st)
                    }
                    (Some(t), SliceOpDir::Front) => {
                        let p = b.tuple_proj(t, 0);
                        let ll = b.tuple_proj(t, 1);
                        let st = b.tuple_proj(t, 2);
                        let mut physical = b.slice_push(p, values.clone(), SliceOpDir::Front);
                        for (k, value) in values.iter().enumerate() {
                            let k_const = b.int_const(32, k as u128);
                            let idx = b.uadd(st, k_const);
                            physical = b.array_set(physical, idx, *value);
                        }
                        (physical, b.uadd(ll, bump), st)
                    }
                    (None, SliceOpDir::Back) => {
                        let base_len = b.slice_len(slice);
                        let mut physical = slice;
                        for value in &values {
                            physical = b.slice_push(physical, vec![*value], SliceOpDir::Back);
                        }
                        let zero = b.int_const(32, 0);
                        (physical, b.uadd(base_len, bump), zero)
                    }
                    (None, SliceOpDir::Front) => {
                        let base_len = b.slice_len(slice);
                        let physical = b.slice_push(slice, values.clone(), SliceOpDir::Front);
                        let zero = b.int_const(32, 0);
                        (physical, b.uadd(base_len, bump), zero)
                    }
                }
            };
            let t = mk_slice_tuple(
                physical, log_len, start, phys_ty, function, ssa, new_instrs, loc,
            );
            replacement_tuple_map.insert(result, t);
        }

        OpCode::SlicePop {
            dir,
            result_slice,
            result_elem,
            slice,
        } if is_window_op(result_slice, slice, affected, replacement_tuple_map) => {
            let phys_ty = physical_type(result_slice, affected, type_info);
            let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone());
            let (p, ll, st) = if let Some(&t) = replacement_tuple_map.get(&slice) {
                (b.tuple_proj(t, 0), b.tuple_proj(t, 1), b.tuple_proj(t, 2))
            } else {
                let ll = b.slice_len(slice);
                let st = b.int_const(32, 0);
                (slice, ll, st)
            };
            let zero = b.int_const(32, 0);
            b.assert_cmp(CmpKind::ULt, zero, ll);
            let one = b.int_const(32, 1);
            let new_ll = b.usub(ll, one);
            let (elem_index, new_st) = match dir {
                SliceOpDir::Back => (b.uadd(st, new_ll), st),
                SliceOpDir::Front => (st, b.uadd(st, one)),
            };
            new_instrs.push(
                OpCode::ArrayGet {
                    result: result_elem,
                    array: p,
                    index: elem_index,
                }
                .locate(loc.clone()),
            );
            let t = mk_slice_tuple(p, new_ll, new_st, phys_ty, function, ssa, new_instrs, loc);
            replacement_tuple_map.insert(result_slice, t);
        }

        OpCode::SliceInsert {
            result,
            slice,
            index,
            value,
        } if is_window_op(result, slice, affected, replacement_tuple_map) => {
            debug_assert_u32_index(index, type_info);
            let phys_ty = physical_type(result, affected, type_info);
            let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone());
            let (p, ll, st) = if let Some(&t) = replacement_tuple_map.get(&slice) {
                (b.tuple_proj(t, 0), b.tuple_proj(t, 1), b.tuple_proj(t, 2))
            } else {
                let ll = b.slice_len(slice);
                let st = b.int_const(32, 0);
                (slice, ll, st)
            };
            let one = b.int_const(32, 1);
            let new_ll = b.uadd(ll, one);
            b.assert_cmp(CmpKind::ULt, index, new_ll);
            let phys_idx = b.uadd(st, index);
            let physical = b.slice_insert(p, phys_idx, value);
            let t = mk_slice_tuple(
                physical, new_ll, st, phys_ty, function, ssa, new_instrs, loc,
            );
            replacement_tuple_map.insert(result, t);
        }

        OpCode::SliceRemove {
            result_slice,
            result_elem,
            slice,
            index,
        } if is_window_op(result_slice, slice, affected, replacement_tuple_map) => {
            debug_assert_u32_index(index, type_info);
            let phys_ty = physical_type(result_slice, affected, type_info);
            let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone());
            let (p, ll, st) = if let Some(&t) = replacement_tuple_map.get(&slice) {
                (b.tuple_proj(t, 0), b.tuple_proj(t, 1), b.tuple_proj(t, 2))
            } else {
                let ll = b.slice_len(slice);
                let st = b.int_const(32, 0);
                (slice, ll, st)
            };
            b.assert_cmp(CmpKind::ULt, index, ll);
            let one = b.int_const(32, 1);
            let new_ll = b.usub(ll, one);
            let phys_idx = b.uadd(st, index);
            let physical = b.fresh_value();
            new_instrs.push(
                OpCode::SliceRemove {
                    result_slice: physical,
                    result_elem,
                    slice: p,
                    index: phys_idx,
                }
                .locate(loc.clone()),
            );
            let t = mk_slice_tuple(
                physical, new_ll, st, phys_ty, function, ssa, new_instrs, loc,
            );
            replacement_tuple_map.insert(result_slice, t);
        }

        OpCode::ArrayGet {
            result,
            array,
            index,
        } if replacement_tuple_map.contains_key(&array) => {
            debug_assert_u32_index(index, type_info);
            let t = replacement_tuple_map[&array];
            let (physical, phys_index) = {
                let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone());
                let p = b.tuple_proj(t, 0);
                let ll = b.tuple_proj(t, 1);
                let st = b.tuple_proj(t, 2);
                b.assert_cmp(CmpKind::ULt, index, ll);
                (p, b.uadd(st, index))
            };
            new_instrs.push(
                OpCode::ArrayGet {
                    result,
                    array: physical,
                    index: phys_index,
                }
                .locate(loc),
            );
        }

        OpCode::ArraySet {
            result,
            array,
            index,
            value,
        } if replacement_tuple_map.contains_key(&array) => {
            debug_assert_u32_index(index, type_info);
            let phys_ty = physical_type(result, affected, type_info);
            let t = replacement_tuple_map[&array];
            let (physical, log_len, start) = {
                let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone());
                let p = b.tuple_proj(t, 0);
                let ll = b.tuple_proj(t, 1);
                let st = b.tuple_proj(t, 2);
                b.assert_cmp(CmpKind::ULt, index, ll);
                let phys_index = b.uadd(st, index);
                (b.array_set(p, phys_index, value), ll, st)
            };
            let t2 = mk_slice_tuple(
                physical, log_len, start, phys_ty, function, ssa, new_instrs, loc,
            );
            replacement_tuple_map.insert(result, t2);
        }

        // `Alloc`/`Store`: the slot and the stored value must agree on window-ness. The
        // `(Some(_), false)` arm is the same refusal the `Call` arm makes for an argument, and for
        // the same reason: without it a window would be written into a cell the joined shapes typed
        // as a bare slice, and the mismatch would surface far away as a `Types` error instead of
        // here.
        OpCode::Alloc { result, value } => {
            let slot_is_window = is_ref_to_wl_slice(result, affected);
            let value = match (replacement_tuple_map.get(&value).copied(), slot_is_window) {
                (Some(t), true) => t,
                (Some(_), false) => panic!(
                    "ICE: purify_witness_slices: slice window initializes an alloc whose joined shapes left the slot pure"
                ),
                (None, true) => {
                    assert!(
                        !affected.contains_key(&value),
                        "ICE: purify_witness_slices: alloc initializer v{} is affected but has no slice tuple",
                        value.0
                    );
                    materialize_pure_slice_tuple(value, type_info, function, ssa, new_instrs)
                }
                (None, false) => value,
            };
            new_instrs.push(OpCode::Alloc { result, value }.locate(loc));
        }

        OpCode::Store { ptr, value } => {
            let slot_is_window = is_ref_to_wl_slice(ptr, affected);
            let value = match (replacement_tuple_map.get(&value).copied(), slot_is_window) {
                (Some(t), true) => t,
                (Some(_), false) => panic!(
                    "ICE: purify_witness_slices: slice window stored into a slot whose joined shapes left it pure"
                ),
                (None, true) => {
                    assert!(
                        !affected.contains_key(&value),
                        "ICE: purify_witness_slices: stored value v{} is affected but has no slice tuple",
                        value.0
                    );
                    materialize_pure_slice_tuple(value, type_info, function, ssa, new_instrs)
                }
                (None, false) => value,
            };
            new_instrs.push(OpCode::Store { ptr, value }.locate(loc));
        }

        OpCode::Load { result, ptr } => {
            new_instrs.push(OpCode::Load { result, ptr }.locate(loc));
            // The slot decides, not the loaded value's own shape: a `Load` aliases whatever the
            // cell holds, so if the cell is a window the result is one — even where this site's
            // shapes call it pure (the same per-site/joined skew `is_window_op` documents).
            if is_ref_to_wl_slice(ptr, affected) || is_wl_slice(result, affected) {
                // The converse must not happen: a wl result over a pure-length slot would alias a
                // bare slice as a tuple. It can only arise from a witness-selected pointer, and
                // Noir rejects those so we can fail loudly instead.
                assert!(
                    is_ref_to_wl_slice(ptr, affected),
                    "ICE: purify_witness_slices: wl slice loaded from a pure-length slot"
                );
                replacement_tuple_map.insert(result, result);
            }
        }

        OpCode::Call {
            results,
            function: callee,
            args,
            unconstrained,
        } => {
            let CallTarget::Static(g) = &callee else {
                panic!("ICE: dynamic call survived to purify_witness_slices")
            };
            let lifted_params = lifts.params.get(g).map(Vec::as_slice).unwrap_or(&[]);
            let args = args
                .into_iter()
                .enumerate()
                .map(|(i, a)| {
                    let lifted = lifted_params.get(i).copied().unwrap_or(false);
                    match (replacement_tuple_map.get(&a).copied(), lifted) {
                        (Some(t), true) => t,
                        (None, true) => {
                            assert!(
                                !affected.contains_key(&a),
                                "ICE: purify_witness_slices: call argument v{} is affected but has no slice tuple",
                                a.0
                            );
                            materialize_pure_slice_tuple(a, type_info, function, ssa, new_instrs)
                        }
                        (Some(_), false) => panic!(
                            "ICE: purify_witness_slices: wl slice tuple flows into a param the \
                             joined shapes left pure"
                        ),
                        (None, false) => a,
                    }
                })
                .collect();

            // `lifts` holds the joined (context-insensitive) signature, while `affected` holds this
            // call site's own shapes, so `lifted && !is_wl_slice(r)` is the ordinary
            // over-approximation. A callee whose return is wl at some *other* call site returns a
            // tuple here too, and registering `r` in the map is exactly what makes that work.
            //
            // The converse cannot happen (the join dominates every site) and would be unsound: a wl
            // result with no tuple behind it. Do not "symmetrise" this into an `assert_eq!`. It is
            // written as is to ensure soundness.
            let lifted_returns = lifts.returns.get(g).map(Vec::as_slice).unwrap_or(&[]);
            for (i, &r) in results.iter().enumerate() {
                if lifted_returns.get(i).copied().unwrap_or(false) {
                    replacement_tuple_map.insert(r, r);
                } else {
                    assert!(
                        !is_wl_slice(r, affected),
                        "ICE: purify_witness_slices: caller-side wl result but the callee's \
                         joined return stayed pure"
                    );
                }
            }
            new_instrs.push(
                OpCode::Call {
                    results,
                    function: callee,
                    args,
                    unconstrained,
                }
                .locate(loc),
            );
        }

        OpCode::Cast {
            result,
            value,
            target,
        } => {
            // Rewriting `value` through the tuple map (and aliasing `result` to a tuple) is only
            // meaningful for `Nop`: any other target would apply the cast to the 3-tuple itself
            // (e.g. `Map` reaches `result_type`'s slice arm and panics).
            let mapped = replacement_tuple_map.get(&value).copied();
            assert!(
                matches!(target, CastTarget::Nop)
                    || (mapped.is_none() && !is_wl_slice(result, affected)),
                "ICE: purify_witness_slices: non-Nop cast ({target:?}) touches a wl slice"
            );
            let value = mapped.unwrap_or(value);
            new_instrs.push(
                OpCode::Cast {
                    result,
                    value,
                    target,
                }
                .locate(loc),
            );

            // Registered on `mapped` as well as on the result's own shape, for the second of
            // [`is_window_op`]'s two reasons: a `Nop` cast of a window *aliases* that window no
            // matter what this site's shapes say about the result. Dropping the alias out of
            // `replacement_tuple_map` would break the forward closure the map *is*, and (unlike
            // every other arm) it would do so silently.
            //
            // A downstream slice op would find neither reason in `is_window_op`, fall through to
            // the `other` fallback whose two asserts both also miss it (the result is in neither
            // `affected` nor the map), and be emitted holding a bare 3-tuple.
            if is_wl_slice(result, affected) || mapped.is_some() {
                replacement_tuple_map.insert(result, result);
            }
        }

        OpCode::Select {
            result,
            cond,
            if_t,
            if_f,
        } => {
            // The condition taints the result's `Len` (`leaf_paths` includes it), so a select
            // between two *pure*-length slices can still be wl. Both arms must then become tuples —
            // the same materialization the `Jmp`/`Return`/`Call` boundaries do — or the select
            // would produce a plain slice while `result` is registered as a tuple.
            //
            // Whichever way it goes, the two arms have to agree, which is why both are keyed on the
            // *result*'s shape rather than on their own. An arm carrying a window while the result
            // shapes call it pure would be passed through as a tuple while its sibling stayed a
            // bare slice, leaving the select ill-typed.
            //
            // Thus [`close_boundaries`] raises the result ahead of time (its `plan_windows` ->
            // `Select` rule) rather than letting it come up here. The refusal is the backstop for a
            // result the closure could not widen, the same role it plays elsewhere.
            let result_is_window = is_wl_slice(result, affected);
            let mut arm = |v: ValueId, function: &mut HLFunction, ssa: &mut HLSSA| match (
                replacement_tuple_map.get(&v).copied(),
                result_is_window,
            ) {
                (Some(t), true) => t,
                (Some(_), false) => panic!(
                    "ICE: purify_witness_slices: select arm v{} carries a slice window but the result's shapes left it pure",
                    v.0
                ),
                (None, true) => {
                    assert!(
                        !affected.contains_key(&v),
                        "ICE: purify_witness_slices: select arm v{} is affected but has no slice tuple",
                        v.0
                    );
                    materialize_pure_slice_tuple(v, type_info, function, ssa, new_instrs)
                }
                (None, false) => v,
            };
            let if_t = arm(if_t, function, ssa);
            let if_f = arm(if_f, function, ssa);

            new_instrs.push(
                OpCode::Select {
                    result,
                    cond,
                    if_t,
                    if_f,
                }
                .locate(loc),
            );
            if is_wl_slice(result, affected) {
                replacement_tuple_map.insert(result, result);
            }
        }

        // Guards do not appear pre-WTI, and this pass should not be the place that starts
        // supporting them: `ElideTuples` runs immediately after it (see the `initial_ssa` pass
        // list) and panics on *any* `Guard` while planning components, so nothing this arm could
        // produce would survive to the next pass. Refuse at the same contract level rather than
        // pretend to handle a shape the pipeline drops on the floor one pass later.
        OpCode::Guard { .. } => {
            panic!("ICE: purify_witness_slices: Guard encountered before witness typing")
        }

        other => {
            assert!(
                !other
                    .get_inputs()
                    .chain(other.get_results())
                    .any(|v| affected.contains_key(v)),
                "purify_witness_slices: witness-length slice flows into an unsupported \
                 opcode: {other:?}"
            );

            // Separate check, because window-ness is not always visible in `affected`: a value
            // lifted by a callee's joined signature is a `(physical, log_len, start)` tuple even
            // where this site's shapes call it pure (see [`is_window_op`]). Without this, such a
            // value would be handed to an op expecting a bare slice and only surface much later as
            // "Type is not an array: Tuple<Slice<..>, u32, u32>" from `Types`.
            assert!(
                !other
                    .get_inputs()
                    .any(|v| replacement_tuple_map.contains_key(v)),
                "purify_witness_slices: slice window flows into an unsupported opcode: {other:?}"
            );
            new_instrs.push(other.locate(loc));
        }
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use mavros_artifacts::FieldConfig;

    use super::*;
    use crate::compiler::{
        Field,
        analysis::{types::Types, witness_info::WitnessType},
        ssa::hlssa::{
            SequenceTargetType,
            builder::{HLBlockEmitter, HLEmitter, HLSSABuilder},
        },
    };

    fn fr(n: u64) -> Field {
        FieldConfig::bn254().constant(n)
    }

    fn wl_tuple_type() -> Type {
        Type::tuple_of(vec![Type::field().slice_of(), Type::int(32), Type::int(32)])
    }

    /// Ops of the (possibly cloned) entry function, in block order.
    fn entry_ops(ssa: &HLSSA) -> Vec<OpCode> {
        ssa.get_function(ssa.get_unique_entrypoint_id())
            .get_blocks()
            .flat_map(|(_, block)| block.get_instructions())
            .cloned()
            .collect()
    }

    /// Whether some block of the entry function has a parameter of the purified tuple type.
    fn has_tuple_param(ssa: &HLSSA) -> bool {
        ssa.get_function(ssa.get_unique_entrypoint_id())
            .get_blocks()
            .any(|(_, block)| block.get_parameters().any(|(_, ty)| *ty == wl_tuple_type()))
    }

    /// The defining op of the entry function's (sole) returned value.
    fn return_def(ssa: &HLSSA) -> OpCode {
        let function = ssa.get_function(ssa.get_unique_entrypoint_id());
        let returned = function
            .get_blocks()
            .find_map(|(_, block)| match block.get_terminator() {
                Some(Terminator::Return(values)) => Some(values[0]),
                _ => None,
            })
            .expect("entry should return a value");
        function
            .get_blocks()
            .flat_map(|(_, block)| block.get_instructions())
            .find(|op| op.get_results().any(|r| *r == returned))
            .expect("returned value should be instruction-defined")
            .clone()
    }

    fn run_pass(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let mut store = AnalysisStore::new();
        store.insert_with_deps::<FlowAnalysis>(flow, vec![]);
        store.insert_with_deps::<TypeInfo>(types, vec![]);
        PurifyWitnessSlices::new().run(ssa, &store);
        // The rewrite must leave well-typed SSA behind.
        let flow = FlowAnalysis::run(ssa);
        let _ = Types::new().run(ssa, &flow);
    }

    /// Build `main(x)`: a diamond on a witness condition whose arms produce pure slices of
    /// lengths 1 and 2, then `body` consumes the merged slice; returns `body`'s value.
    fn build_differing_length_merge(
        ssa: &mut HLSSA,
        body: impl FnOnce(&mut HLBlockEmitter<'_>, ValueId) -> ValueId,
    ) {
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::int(32));
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            let zero = e.field_const(fr(0));
            let cond = e.eq(w, zero);
            let merge = e.build_if_else(
                cond,
                vec![Type::field().slice_of()],
                |e| {
                    let c = e.field_const(fr(7));
                    vec![e.mk_seq(vec![c], SequenceTargetType::Slice, Type::field())]
                },
                |e| {
                    let c1 = e.field_const(fr(1));
                    let c2 = e.field_const(fr(2));
                    vec![e.mk_seq(vec![c1, c2], SequenceTargetType::Slice, Type::field())]
                },
            );
            let r = body(&mut e, merge[0]);
            e.terminate_return(vec![r]);
        });
    }

    /// Build the per-site/joined skew in an SSA of two functions.
    ///
    /// `extend(s) = s.push_back(7)` is called twice: once *inside* a witness branch, where this
    /// site's shapes still call the result pure, and once on the merged slice afterwards. It is
    /// that second site that lifts `extend`'s *joined* return to a window, so the callee returns a
    /// `(physical, log_len, start)` tuple at **both**. The in-branch result handed to `consume` is
    /// therefore a window that `affected` calls pure: exactly the shape [`is_window_op`] exists
    /// for, and the only way to reach the arms keyed on `affected` alone.
    ///
    /// `consume` returns the slice its branch merges, so it can either transform the window or
    /// just use it for effect and hand it straight back.
    fn build_joined_lift_skew(
        ssa: &mut HLSSA,
        consume: impl FnOnce(&mut HLBlockEmitter<'_>, ValueId) -> ValueId,
    ) {
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(ssa);
        let extend_id = sb.ssa().add_function("extend".to_string());

        sb.modify_function(extend_id, |b| {
            b.function.add_return_type(Type::field().slice_of());
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let s = e.add_parameter(Type::field().slice_of());
            let c7 = e.field_const(fr(7));
            let pushed = e.slice_push(s, vec![c7], SliceOpDir::Back);
            e.terminate_return(vec![pushed]);
        });

        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::int(32));
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            let zero = e.field_const(fr(0));
            let cond = e.eq(w, zero);
            let c1 = e.field_const(fr(1));
            let s = e.mk_seq(vec![c1], SequenceTargetType::Slice, Type::field());

            let merge = e.build_if_else(
                cond,
                vec![Type::field().slice_of()],
                |e| {
                    let t = e.call(extend_id, vec![s], 1)[0];
                    vec![consume(e, t)]
                },
                |_e| vec![s],
            );

            // The wl-argument site: this is what lifts `extend`'s joined return.
            let again = e.call(extend_id, vec![merge[0]], 1)[0];
            let len = e.slice_len(again);
            e.terminate_return(vec![len]);
        });
    }

    /// Build the boundary skew in three functions.
    ///
    /// `extend(s) = s.push_back(7)` is called with a witness-length argument in `main`, which lifts
    /// its *joined* parameter and return. `mid(s) = extend(s)` calls it at a site whose own shapes
    /// still call everything pure-length, so `mid` ends up holding — and returning — a window that
    /// neither its own return shape nor its caller's shapes know about.
    ///
    /// `plumb` decides how `mid` gets its slice: by value, or through a `&mut` cell that `main`'s
    /// own shapes leave pure-length. Both are boundaries a window has to cross with nothing to
    /// materialize on the far side.
    fn build_boundary_skew(ssa: &mut HLSSA, through_ref: bool) {
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(ssa);
        let extend_id = sb.ssa().add_function("extend".to_string());
        let mid_id = sb.ssa().add_function("mid".to_string());

        sb.modify_function(extend_id, |b| {
            b.function.add_return_type(Type::field().slice_of());
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let s = e.add_parameter(Type::field().slice_of());
            let c7 = e.field_const(fr(7));
            let pushed = e.slice_push(s, vec![c7], SliceOpDir::Back);
            e.terminate_return(vec![pushed]);
        });

        // `mid` returns `extend`'s result: a window flowing out through a return slot this
        // function's own shapes call pure. Through a ref, it writes it back into the caller's cell.
        sb.modify_function(mid_id, |b| {
            b.function.add_return_type(Type::field().slice_of());
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            if through_ref {
                let r = e.add_parameter(Type::field().slice_of().ref_of());
                let loaded = e.load(r);
                let t = e.call(extend_id, vec![loaded], 1)[0];
                e.store(r, t);
                e.terminate_return(vec![t]);
            } else {
                let s = e.add_parameter(Type::field().slice_of());
                let t = e.call(extend_id, vec![s], 1)[0];
                e.terminate_return(vec![t]);
            }
        });

        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::int(32));
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            let zero = e.field_const(fr(0));
            let cond = e.eq(w, zero);
            let c1 = e.field_const(fr(1));
            let s = e.mk_seq(vec![c1], SequenceTargetType::Slice, Type::field());

            // The pure-length site.
            let pure_result = if through_ref {
                let cell = e.alloc(s);
                let out = e.call(mid_id, vec![cell], 1)[0];
                let _ = e.load(cell);
                out
            } else {
                e.call(mid_id, vec![s], 1)[0]
            };

            // The witness-length site: this is what lifts `extend`'s joined signature.
            let merge = e.build_if_else(
                cond,
                vec![Type::field().slice_of()],
                |e| vec![e.slice_push(s, vec![c1], SliceOpDir::Back)],
                |_e| vec![s],
            );
            let again = e.call(extend_id, vec![merge[0]], 1)[0];

            let pure_len = e.slice_len(pure_result);
            let wl_len = e.slice_len(again);
            let total = e.uadd(pure_len, wl_len);
            e.terminate_return(vec![total]);
        });
    }

    /// A window leaving through a return slot the function's own shapes call pure.
    ///
    /// `is_window_op` carries the joined lift *forward* through the ops that consume the window,
    /// but a return slot is not an op: before `close_boundaries` raised `mid`'s slot too, the
    /// window reached `Return` and hit "wl slice tuple flows into a pure".
    #[test]
    fn window_out_of_a_pure_return_slot_lifts_the_slot() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_boundary_skew(&mut ssa, false);
        run_pass(&mut ssa);

        // Both `extend` (lifted by its own shapes) and `mid` (lifted only by the closure) must now
        // return the window.
        let lifted = ssa
            .get_function_ids()
            .filter(|f| ssa.get_function(*f).get_returns() == [wl_tuple_type()])
            .count();
        assert_eq!(
            lifted, 2,
            "the callee's lift must travel out through its caller's return slot"
        );
    }

    /// The `&mut` counterpart, and the one with no way out other than widening: the callee writes
    /// through the pointer, so a copy-out shim at the boundary would have to turn a window back
    /// into a witness-length slice. The caller's own cell has to become a window instead — free,
    /// since a pure-length slice held as one is just `(physical, physical.len(), 0)`.
    ///
    /// Before the closure, the callee's parameter was retyped while the caller passed its cell
    /// unchanged, and `ElideTuples` split only one side: the call arrived with the wrong arity and
    /// crashed `Types` with a bare "Error running opcode Call { .. }".
    #[test]
    fn window_through_a_pure_ref_cell_widens_the_caller() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_boundary_skew(&mut ssa, true);
        run_pass(&mut ssa);

        let main = ssa.get_function(ssa.get_unique_entrypoint_id());
        let allocs_a_window = main
            .get_blocks()
            .flat_map(|(_, block)| block.get_instructions())
            .any(|op| matches!(op, OpCode::MkTuple { .. }));
        assert!(
            allocs_a_window,
            "the caller's pure-length cell must be initialized with a materialized window"
        );
    }

    /// A window reaching a `Nop` cast must stay a window.
    ///
    /// The cast aliases the tuple, so the alias has to be registered on the strength of its operand
    /// as the result's own shape calls it pure here (see [`build_joined_lift_skew`]).
    #[test]
    fn window_through_nop_cast_stays_a_window() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_joined_lift_skew(&mut ssa, |e, t| {
            let aliased = e.cast_to(CastTarget::Nop, t);
            let c2 = e.field_const(fr(2));
            e.slice_push(aliased, vec![c2], SliceOpDir::Back)
        });
        run_pass(&mut ssa);

        // The push through the alias must have taken the windowed form, whose signature is the
        // `ArraySet` that writes the value at the witness cursor.
        let ops = entry_ops(&ssa);
        assert!(
            ops.iter().any(|op| matches!(op, OpCode::ArraySet { .. })),
            "the push through the cast alias must be rewritten as a window push"
        );
    }

    /// The `Alloc` counterpart: a window initializing a cell that this function's shapes left
    /// pure-length. Typing the cell as a bare slice would surface far away in `Types`, so
    /// `close_boundaries` widens the cell to match the value instead — a pure-length slice held as
    /// a window costs nothing. The arm's refusal survives only as the backstop for a cell whose
    /// aliases this pass cannot reach.
    ///
    /// `run_pass` re-runs `Types` over the result, so type-checking is the assertion.
    #[test]
    fn window_into_a_pure_alloc_slot_widens_the_cell() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_joined_lift_skew(&mut ssa, |e, t| {
            let p = e.alloc(t);
            e.load(p)
        });
        run_pass(&mut ssa);

        assert!(
            entry_ops(&ssa)
                .iter()
                .any(|op| matches!(op, OpCode::Alloc { .. })),
            "the alloc must survive, now holding the window"
        );
    }

    /// The `Select` counterpart: an arm carrying a window while the result's shapes call it pure
    /// would be passed through as a tuple while its sibling stayed a bare slice, leaving the select
    /// itself ill-typed. Unlike the boundaries above, a select cannot give way on the side the
    /// window arrives from — its two alternatives have to agree — so `close_boundaries` widens the
    /// *result*, and the pure sibling materializes a window like every other boundary does.
    ///
    /// `run_pass` re-runs `Types` over the result, so type-checking is half the assertion.
    #[test]
    fn window_select_arm_widens_the_pure_result() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_joined_lift_skew(&mut ssa, |e, t| {
            let c1 = e.field_const(fr(1));
            let s = e.mk_seq(vec![c1], SequenceTargetType::Slice, Type::field());
            let pure_cond = e.eq(c1, c1);
            e.select(pure_cond, t, s)
        });
        run_pass(&mut ssa);

        // The select must survive choosing between two windows: the lifted callee's, and one
        // materialized for the pure arm.
        let ops = entry_ops(&ssa);
        assert!(
            ops.iter().any(|op| matches!(op, OpCode::Select { .. })),
            "the select itself must survive"
        );
        assert!(
            ops.iter().any(|op| matches!(op, OpCode::MkTuple { .. })),
            "the pure select arm must materialize a window to match its sibling"
        );
    }

    /// A witness-length slice nested in a container has no rewrite anywhere in this pass, and the
    /// block-parameter path would retype such a parameter without converting its jump arguments.
    /// Refuse where the shape is minted instead.
    #[test]
    #[should_panic(expected = "witness-length slice nested in a container")]
    fn window_nested_in_a_container_is_an_ice() {
        let inner = WitnessShape::Slice(
            WitnessType::Witness,
            Box::new(WitnessShape::Scalar(WitnessType::Pure)),
        );
        let shape = WitnessShape::Slice(WitnessType::Pure, Box::new(inner));
        let _ = purify_type(&Type::field().slice_of().slice_of(), &shape);
    }

    /// The control for the test above: a window *behind a `Ref`* is the ordinary `&mut [Field]`
    /// case and must keep purifying, since a ref's pointee type is inferred from its `Alloc` and
    /// tracks the rewrite on its own.
    #[test]
    fn window_behind_a_ref_still_purifies() {
        let shape = WitnessShape::Ref(
            WitnessType::Pure,
            Box::new(WitnessShape::Slice(
                WitnessType::Witness,
                Box::new(WitnessShape::Scalar(WitnessType::Pure)),
            )),
        );
        let purified = purify_type(&Type::field().slice_of().ref_of(), &shape);
        assert_eq!(purified, wl_tuple_type().ref_of());
    }

    /// The soundness-critical case: two pure slices of different static lengths merged under a
    /// witness condition, with no slice ops anywhere. The pass must still fire: the merge
    /// parameter becomes a `(physical, log_len, start)` tuple, both arms materialize tuples for
    /// their pure jump arguments, and `SliceLen` reads `log_len` via projection.
    #[test]
    fn differing_length_merge_without_pushes_purifies_the_merge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_differing_length_merge(&mut ssa, |e, merged| e.slice_len(merged));
        run_pass(&mut ssa);

        assert!(
            has_tuple_param(&ssa),
            "merge param should become the wl tuple"
        );
        // The returned length must read log_len off the tuple, not a surviving SliceLen. (The
        // arm materializations legitimately keep SliceLen ops on the pure physical slices.)
        assert!(
            matches!(return_def(&ssa), OpCode::TupleProj { idx: 1, .. }),
            "the length read must project log_len"
        );
        let ops = entry_ops(&ssa);
        let mk_tuples = ops
            .iter()
            .filter(|op| matches!(op, OpCode::MkTuple { .. }))
            .count();
        assert!(
            mk_tuples >= 2,
            "both pure arms must materialize wl tuples, found {mk_tuples}"
        );
    }

    /// A push guarded by a witness branch: the in-branch push result is still pure-shaped, so
    /// the push op itself survives untouched; wl-ness materializes at the merge.
    #[test]
    fn guarded_push_survives_pure_and_len_projects() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::int(32));
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            let zero = e.field_const(fr(0));
            let cond = e.eq(w, zero);
            let c1 = e.field_const(fr(1));
            let s = e.mk_seq(vec![c1], SequenceTargetType::Slice, Type::field());
            let merge = e.build_if_else(
                cond,
                vec![Type::field().slice_of()],
                |e| {
                    let c2 = e.field_const(fr(2));
                    vec![e.slice_push(s, vec![c2], SliceOpDir::Back)]
                },
                |_e| vec![s],
            );
            let len = e.slice_len(merge[0]);
            e.terminate_return(vec![len]);
        });
        run_pass(&mut ssa);

        assert!(
            has_tuple_param(&ssa),
            "merge param should become the wl tuple"
        );
        let ops = entry_ops(&ssa);
        assert!(
            ops.iter().any(|op| matches!(op, OpCode::SlicePush { .. })),
            "the pure in-branch push must survive as a plain push"
        );
        assert!(
            matches!(return_def(&ssa), OpCode::TupleProj { idx: 1, .. }),
            "the length read must project log_len"
        );
    }

    /// A push onto an already-witness-length slice: the rewrite grows the physical slice and
    /// writes the value at the witness cursor `start + log_len` via `ArraySet`.
    #[test]
    fn push_onto_wl_slice_writes_at_witness_cursor() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_differing_length_merge(&mut ssa, |e, merged| {
            let c9 = e.field_const(fr(9));
            let pushed = e.slice_push(merged, vec![c9], SliceOpDir::Back);
            e.slice_len(pushed)
        });
        run_pass(&mut ssa);

        let ops = entry_ops(&ssa);
        assert!(
            ops.iter().any(|op| matches!(op, OpCode::ArraySet { .. })),
            "wl back-push must write the value at the witness cursor"
        );
        assert!(
            matches!(return_def(&ssa), OpCode::TupleProj { idx: 1, .. }),
            "the length of the pushed wl slice must project log_len"
        );
    }

    /// A `Guard` reaching this pass is an ICE, not a shape to rewrite. It cannot be supported here
    /// even in principle: `ElideTuples` runs immediately afterwards and panics on any `Guard` while
    /// planning components, so a rewrite would only move the crash one pass later. This test pins
    /// the refusal so the two passes' contracts don't diverge.
    #[test]
    #[should_panic(expected = "Guard encountered before witness typing")]
    fn guard_before_witness_typing_is_an_ice() {
        let mut ssa = HLSSA::with_main("main".to_string());
        build_differing_length_merge(&mut ssa, |e, merged| {
            let c9 = e.field_const(fr(9));
            let zero = e.field_const(fr(0));
            let gcond = e.eq(c9, zero);
            let pushed = e.fresh_value();
            e.emit(OpCode::Guard {
                condition: gcond,
                inner: Box::new(OpCode::SlicePush {
                    dir: SliceOpDir::Back,
                    result: pushed,
                    slice: merged,
                    values: vec![c9],
                }),
            });
            e.slice_len(pushed)
        });
        run_pass(&mut ssa);
    }

    /// A witness `Select` between two *pure* slices of different lengths: the condition taints the
    /// result's `Len`, so the select itself is wl and both arms must materialize tuples — the
    /// single-instruction analogue of `differing_length_merge_without_pushes_purifies_the_merge`.
    ///
    /// Noir lowers `if` on slices to a cfg diamond, and `untaint_control_flow` (the only producer
    /// of slice selects) runs *after* this pass, so — like the `Guard` arm above — this arm is
    /// reachable only from a hand-built program and this test is its only coverage.
    #[test]
    fn wl_select_materializes_tuples_for_both_arms() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::int(32));
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let x = e.add_parameter(Type::field());
            let w = e.write_witness(x);
            let zero = e.field_const(fr(0));
            let cond = e.eq(w, zero);
            let c1 = e.field_const(fr(1));
            let c2 = e.field_const(fr(2));
            let s1 = e.mk_seq(vec![c1], SequenceTargetType::Slice, Type::field());
            let s2 = e.mk_seq(vec![c1, c2], SequenceTargetType::Slice, Type::field());
            let m = e.select(cond, s1, s2);
            let len = e.slice_len(m);
            e.terminate_return(vec![len]);
        });
        run_pass(&mut ssa);

        let ops = entry_ops(&ssa);
        let mk_tuples = ops
            .iter()
            .filter(|op| matches!(op, OpCode::MkTuple { .. }))
            .count();
        assert!(
            mk_tuples >= 2,
            "both pure select arms must materialize wl tuples, found {mk_tuples}"
        );
        assert!(
            ops.iter().any(|op| matches!(op, OpCode::Select { .. })),
            "the select itself must survive, now choosing between the two tuples"
        );
        assert!(
            matches!(return_def(&ssa), OpCode::TupleProj { idx: 1, .. }),
            "the length read must project log_len"
        );
    }
}
