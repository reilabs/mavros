//! Implements a defunctionalization pass that eliminates first-class function values from the SSA,
//! ensuring that the rest of the compiler only has to deal with calls that can be resolved
//! statically.
//!
//! It does this as follows:
//!
//! 1. Analyses the whole program to determine which functions are reachable from each call target.
//! 2. Synthesizes dispatch functions for each call target that takes the function identifier and
//!    params, dispatches to the right call, and then merges into a single return.
//! 3. Rewrites the original call site to use this dispatch function.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        pass_manager::{AnalysisStore, Pass},
        passes::shared::value_replacements::ValueReplacements,
        ssa::{
            BlockId, FunctionId, Located, SourceLocation, Terminator, ValueId,
            hlssa::{
                CallTarget, Constant, HLSSA, OpCode, Type, TypeExpr,
                builder::{HLEmitter, HLSSABuilder},
            },
        },
    },
};

pub struct Defunctionalize {}

impl Defunctionalize {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for Defunctionalize {
    fn name(&self) -> &'static str {
        "defunctionalize"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        run_defunctionalize(ssa);
    }
}

/// For each SSA value that may hold a function pointer, the set of
/// concrete FunctionIds it can point to.
type ReachingFns = HashMap<(FunctionId, ValueId), HashSet<FunctionId>>;

fn run_defunctionalize(ssa: &mut HLSSA) {
    // Check if there are any FnPtrs at all
    let has_fn_ptrs = ssa
        .const_snapshot()
        .values()
        .any(|cv| matches!(cv.as_ref(), Constant::FnPtr(_)));
    if !has_fn_ptrs {
        return;
    }

    // Phase 1: Compute reaching definitions — which FnPtrs can reach each value
    let reaching = compute_reaching_fn_ptrs(ssa);

    // Phase 2: For each dynamic call site, build a dispatch function
    // with exactly the reachable targets
    let mut call_site_dispatch: HashMap<(FunctionId, ValueId), FunctionId> = HashMap::default();
    let mut dispatch_counter = 0u32;

    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    // Collect all call sites first, then build dispatch functions
    let mut call_sites: Vec<(FunctionId, ValueId)> = Vec::new();
    for &fid in &func_ids {
        let func = ssa.get_function(fid);
        for (_bid, block) in func.get_blocks() {
            for instr in block.get_instructions() {
                if let OpCode::Call {
                    function: CallTarget::Dynamic(fn_ptr_val),
                    ..
                } = instr
                {
                    call_sites.push((fid, *fn_ptr_val));
                }
            }
        }
    }

    for (fid, fn_ptr_val) in &call_sites {
        // Skip if we already built a dispatch for this exact (func, value) pair
        if call_site_dispatch.contains_key(&(*fid, *fn_ptr_val)) {
            continue;
        }
        let targets: Vec<FunctionId> = reaching
            .get(&(*fid, *fn_ptr_val))
            .unwrap_or_else(|| panic!("No reaching FnPtrs for v{} in {:?}", fn_ptr_val.0, fid))
            .iter()
            .copied()
            .collect();

        assert!(
            !targets.is_empty(),
            "Empty target set for v{} in {:?}",
            fn_ptr_val.0,
            fid
        );

        // Get param/return types from the first target (all must match)
        let representative = ssa.get_function(targets[0]);
        let param_types = representative.get_param_types();
        let return_types = representative.get_returns().to_vec();

        let dispatch_fn_id =
            build_dispatch_function(ssa, dispatch_counter, &param_types, &return_types, &targets);
        call_site_dispatch.insert((*fid, *fn_ptr_val), dispatch_fn_id);
        dispatch_counter += 1;
    }

    // Phase 3: Transformation

    // 3a. Intern a `U(32, fn_id)` constant for every FnPtr in storage, build a remap from each
    // FnPtr `ValueId` to its canonical U-typed `ValueId`, and remove the FnPtr entries. The
    // remap is applied globally in phase 3d below, after `Call::Dynamic` rewriting in 3b has
    // run on the still-original operands.
    let mut fnptr_entries: Vec<(ValueId, FunctionId)> = ssa
        .const_snapshot()
        .iter()
        .filter_map(|(vid, cv)| match cv.as_ref() {
            Constant::FnPtr(fn_id) => Some((*vid, *fn_id)),
            _ => None,
        })
        .collect();
    // Sort so the canonical-constant interning order (and thus the assigned ValueIds) does not
    // depend on the snapshot's iteration order.
    fnptr_entries.sort_by_key(|(vid, _)| vid.0);
    let mut fnptr_remap = ValueReplacements::new();
    for (fnptr_vid, fn_id) in &fnptr_entries {
        let canon = ssa.add_const(Constant::U(32, fn_id.0 as u128));
        fnptr_remap.insert(*fnptr_vid, canon);
    }

    // 3b. Replace CallTarget::Dynamic → CallTarget::Static(dispatch_fn). The lookup uses the
    // original FnPtr vid (still present in the IR at this point); the emitted dispatcher Call
    // carries `fn_ptr_val` as args[0], which is remapped to the canonical vid by phase 3d.
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    for fid in func_ids {
        let func = ssa.get_function(fid);
        let has_dynamic = func.get_blocks().any(|(_, block)| {
            block.get_instructions().any(|instr| {
                matches!(
                    instr,
                    OpCode::Call {
                        function: CallTarget::Dynamic(_),
                        ..
                    }
                )
            })
        });
        if !has_dynamic {
            continue;
        }

        let func = ssa.get_function(fid);
        let block_ids: Vec<BlockId> = func.get_blocks().map(|(bid, _)| *bid).collect();
        let func = ssa.get_function_mut(fid);
        for bid in block_ids {
            let block = func.get_block_mut(bid);
            let instructions = block.take_instructions();
            let mut new_instructions = Vec::new();

            for instr in instructions {
                let location = instr.location().clone();
                match instr.payload() {
                    OpCode::Call {
                        results,
                        function: CallTarget::Dynamic(fn_ptr_val),
                        args,
                        unconstrained,
                    } => {
                        let dispatch_fn = *call_site_dispatch
                            .get(&(fid, fn_ptr_val))
                            .unwrap_or_else(|| {
                                panic!("No dispatch function for v{} in {:?}", fn_ptr_val.0, fid)
                            });
                        let mut new_args = Vec::with_capacity(args.len() + 1);
                        new_args.push(fn_ptr_val);
                        new_args.extend(args);

                        new_instructions.push(Located::new(
                            OpCode::Call {
                                results,
                                function: CallTarget::Static(dispatch_fn),
                                args: new_args,
                                unconstrained,
                            },
                            location,
                        ));
                    }
                    other => new_instructions.push(Located::new(other, location)),
                }
            }

            let block = func.get_block_mut(bid);
            block.put_instructions(new_instructions);
        }
    }

    // 3d. Apply the FnPtr → U(32, ...) operand remap globally. This is the operand-rewriting
    // step that used to be implicit when the FnPtr entry was rebound in place at the same vid.
    for (_, func) in ssa.iter_functions_mut() {
        for (_, block) in func.get_blocks_mut() {
            for instr in block.get_instructions_mut() {
                fnptr_remap.replace_inputs(instr);
            }
            fnptr_remap.replace_terminator(block.get_terminator_mut());
        }
    }

    // 3c. Replace TypeExpr::Function → TypeExpr::U(32) everywhere
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    for fid in func_ids {
        let func = ssa.get_function_mut(fid);

        let mut returns = func.take_returns();
        for ret_type in returns.iter_mut() {
            replace_function_type(ret_type);
        }
        for ret_type in returns {
            func.add_return_type(ret_type);
        }

        let block_ids: Vec<BlockId> = func.get_blocks().map(|(bid, _)| *bid).collect();
        for bid in block_ids {
            let block = func.get_block_mut(bid);

            let mut params = block.take_parameters();
            for (_vid, typ) in params.iter_mut() {
                replace_function_type(typ);
            }
            block.put_parameters(params);

            let mut instructions = block.take_instructions();
            for instr in instructions.iter_mut() {
                replace_function_types_in_instruction(&mut *instr);
            }
            block.put_instructions(instructions);
        }
    }
}

/// Compute, for each (function, value) pair, the set of FunctionIds that
/// the value can point to. Uses fixpoint iteration over:
///   - FnPtr constants (seeds)
///   - Jmp block-parameter edges (intra-function)
///   - Static call argument edges (caller → callee params)
///   - Static call return edges (callee returns → caller results)
///   - Dynamic call return edges (resolved target returns → caller results)
fn compute_reaching_fn_ptrs(ssa: &HLSSA) -> ReachingFns {
    let mut reaching: ReachingFns = HashMap::default();
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

    // Check if a type contains Function anywhere (for alias-aware propagation)
    fn contains_function(typ: &Type) -> bool {
        match &typ.expr {
            TypeExpr::Function => true,
            TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) | TypeExpr::Ref(inner) => {
                contains_function(inner)
            }
            TypeExpr::Tuple(elems) => elems.iter().any(contains_function),
            TypeExpr::Field
            | TypeExpr::U(_)
            | TypeExpr::I(_)
            | TypeExpr::WitnessOf(_)
            | TypeExpr::Blob(..) => false,
        }
    }

    // Pre-compute which values are Refs containing Functions (need bidirectional propagation)
    // These come from: Alloc results, block parameters with Ref<...Function...> type
    let mut is_ref_with_fn: HashSet<(FunctionId, ValueId)> = HashSet::default();
    for &fid in &func_ids {
        let func = ssa.get_function(fid);
        for (_bid, block) in func.get_blocks() {
            for instr in block.get_instructions() {
                if let OpCode::Alloc { result, .. } = instr {
                    is_ref_with_fn.insert((fid, *result));
                }
            }
        }
        // Check block parameters
        for (_bid, block) in func.get_blocks() {
            for (vid, typ) in block.get_parameters() {
                if let TypeExpr::Ref(inner) = &typ.expr {
                    if contains_function(inner) {
                        is_ref_with_fn.insert((fid, *vid));
                    }
                }
            }
        }
    }

    // Seed from FnPtr constants in storage. They are module-level — visible to every function
    // that references them — so seed under each function id.
    let fnptr_constants: Vec<(ValueId, FunctionId)> = ssa
        .const_snapshot()
        .iter()
        .filter_map(|(vid, cv)| match cv.as_ref() {
            Constant::FnPtr(fn_id) => Some((*vid, *fn_id)),
            _ => None,
        })
        .collect();
    for &fid in &func_ids {
        for (vid, target) in &fnptr_constants {
            reaching.entry((fid, *vid)).or_default().insert(*target);
        }
    }

    // Pre-compute return values per function
    let mut return_values: HashMap<FunctionId, Vec<ValueId>> = HashMap::default();
    for &fid in &func_ids {
        let func = ssa.get_function(fid);
        for (_bid, block) in func.get_blocks() {
            // It is safe to stop here as Noir doesn't have early return.
            if let Some(Terminator::Return(vals)) = block.get_terminator() {
                return_values.insert(fid, vals.clone());
                break;
            }
        }
    }

    // Helper: merge src set into dest, return true if anything new was added
    fn propagate(
        reaching: &mut ReachingFns,
        src: (FunctionId, ValueId),
        dest: (FunctionId, ValueId),
    ) -> bool {
        let Some(sources) = reaching.get(&src).cloned() else {
            return false;
        };
        let dest_set = reaching.entry(dest).or_default();
        let mut did_change = false;
        for t in sources {
            if dest_set.insert(t) {
                did_change = true;
            }
        }
        did_change
    }

    // Track fn_ptrs stored in global slots (keyed by global offset)
    let mut global_slots: HashMap<usize, HashSet<FunctionId>> = HashMap::default();

    // Fixpoint: propagate reaching sets through edges
    // Backward propagation only happens when source is_ref_with_fn (pointer aliasing)
    let mut changed = true;
    while changed {
        changed = false;
        for &fid in &func_ids {
            let func = ssa.get_function(fid);

            for (_bid, block) in func.get_blocks() {
                if let Some(Terminator::Jmp(dest, args)) = block.get_terminator() {
                    let dest_params: Vec<ValueId> = func
                        .get_block(*dest)
                        .get_parameters()
                        .map(|(vid, _)| *vid)
                        .collect();
                    for (i, arg) in args.iter().enumerate() {
                        if i < dest_params.len() {
                            changed |= propagate(&mut reaching, (fid, *arg), (fid, dest_params[i]));
                            if is_ref_with_fn.contains(&(fid, dest_params[i])) {
                                changed |=
                                    propagate(&mut reaching, (fid, dest_params[i]), (fid, *arg));
                            }
                        }
                    }
                }

                for instr in block.get_instructions() {
                    match instr {
                        OpCode::Call {
                            function,
                            args,
                            results,
                            unconstrained: _,
                        } => {
                            let target_fns: Vec<FunctionId> = match function {
                                CallTarget::Static(callee_id) => vec![*callee_id],
                                CallTarget::Dynamic(fn_ptr_val) => reaching
                                    .get(&(fid, *fn_ptr_val))
                                    .cloned()
                                    .unwrap_or_default()
                                    .into_iter()
                                    .collect(),
                            };
                            for target_fn in target_fns {
                                let callee = ssa.get_function(target_fn);
                                let callee_params: Vec<ValueId> = callee
                                    .get_entry()
                                    .get_parameters()
                                    .map(|(vid, _)| *vid)
                                    .collect();
                                for (i, arg) in args.iter().enumerate() {
                                    if i < callee_params.len() {
                                        changed |= propagate(
                                            &mut reaching,
                                            (fid, *arg),
                                            (target_fn, callee_params[i]),
                                        );
                                        if is_ref_with_fn.contains(&(target_fn, callee_params[i])) {
                                            changed |= propagate(
                                                &mut reaching,
                                                (target_fn, callee_params[i]),
                                                (fid, *arg),
                                            );
                                        }
                                    }
                                }
                                if let Some(ret_vals) = return_values.get(&target_fn) {
                                    for (i, ret_val) in ret_vals.iter().enumerate() {
                                        if i < results.len() {
                                            changed |= propagate(
                                                &mut reaching,
                                                (target_fn, *ret_val),
                                                (fid, results[i]),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        OpCode::MkTuple { result, elems, .. } => {
                            for elem in elems {
                                changed |= propagate(&mut reaching, (fid, *elem), (fid, *result));
                                if is_ref_with_fn.contains(&(fid, *elem)) {
                                    changed |=
                                        propagate(&mut reaching, (fid, *result), (fid, *elem));
                                }
                            }
                        }
                        OpCode::MkSeq { result, elems, .. } => {
                            for elem in elems {
                                changed |= propagate(&mut reaching, (fid, *elem), (fid, *result));
                                if is_ref_with_fn.contains(&(fid, *elem)) {
                                    changed |=
                                        propagate(&mut reaching, (fid, *result), (fid, *elem));
                                }
                            }
                        }
                        OpCode::MkRepeated {
                            result, element, ..
                        } => {
                            changed |= propagate(&mut reaching, (fid, *element), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *element)) {
                                changed |=
                                    propagate(&mut reaching, (fid, *result), (fid, *element));
                            }
                        }
                        OpCode::ArrayGet { result, array, .. } => {
                            changed |= propagate(&mut reaching, (fid, *array), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *result)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *array));
                            }
                        }
                        OpCode::ArraySet {
                            result,
                            array,
                            value,
                            ..
                        } => {
                            changed |= propagate(&mut reaching, (fid, *array), (fid, *result));
                            changed |= propagate(&mut reaching, (fid, *value), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *value)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *array));
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *value));
                            }
                        }
                        OpCode::TupleProj { result, tuple, .. }
                        | OpCode::TupleRefProj {
                            result,
                            tuple_ref: tuple,
                            ..
                        } => {
                            changed |= propagate(&mut reaching, (fid, *tuple), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *result)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *tuple));
                            }
                        }
                        OpCode::Load { result, ptr } => {
                            changed |= propagate(&mut reaching, (fid, *ptr), (fid, *result));
                        }
                        OpCode::Alloc { result, value, .. } => {
                            changed |= propagate(&mut reaching, (fid, *value), (fid, *result));
                        }
                        OpCode::Store { ptr, value } => {
                            changed |= propagate(&mut reaching, (fid, *value), (fid, *ptr));
                        }
                        OpCode::SlicePush {
                            result,
                            slice,
                            values,
                            ..
                        } => {
                            changed |= propagate(&mut reaching, (fid, *slice), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *slice)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *slice));
                            }
                            for v in values {
                                changed |= propagate(&mut reaching, (fid, *v), (fid, *result));
                                if is_ref_with_fn.contains(&(fid, *v)) {
                                    changed |= propagate(&mut reaching, (fid, *result), (fid, *v));
                                }
                            }
                        }
                        OpCode::Select {
                            result, if_t, if_f, ..
                        } => {
                            changed |= propagate(&mut reaching, (fid, *if_t), (fid, *result));
                            changed |= propagate(&mut reaching, (fid, *if_f), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *if_t)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *if_t));
                            }
                            if is_ref_with_fn.contains(&(fid, *if_f)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *if_f));
                            }
                        }
                        OpCode::InitGlobal { global, value } => {
                            if let Some(targets) = reaching.get(&(fid, *value)).cloned() {
                                let slot = global_slots.entry(*global).or_default();
                                for t in targets {
                                    if slot.insert(t) {
                                        changed = true;
                                    }
                                }
                            }
                        }
                        OpCode::ReadGlobal { result, offset, .. } => {
                            if let Some(targets) = global_slots.get(&(*offset as usize)).cloned() {
                                let dest = reaching.entry((fid, *result)).or_default();
                                for t in targets {
                                    if dest.insert(t) {
                                        changed = true;
                                    }
                                }
                            }
                        }
                        // TODO Make exhaustive (#175).
                        _ => {}
                    }
                }
            }
        }
    }

    reaching
}

/// Build a dispatch function for a specific call site's reachable targets.
fn build_dispatch_function(
    ssa: &mut HLSSA,
    counter: u32,
    param_types: &[Type],
    return_types: &[Type],
    variants: &[FunctionId],
) -> FunctionId {
    let name = format!("apply_dispatch@{counter}");
    let location = SourceLocation::synthetic(&name);
    let dispatch_fn_id = ssa.add_function(name);
    let mut sb = HLSSABuilder::new(ssa);
    sb.modify_function(dispatch_fn_id, |b| {
        for ret_type in return_types {
            b.function.add_return_type(ret_type.clone());
        }

        let entry_block = b.function.get_entry_id();

        let fn_id_param;
        let mut forwarded_params: Vec<ValueId> = Vec::new();
        {
            let mut entry = b.block(entry_block);
            fn_id_param = entry.add_parameter(Type::u32());
            for param_type in param_types {
                forwarded_params.push(entry.add_parameter(param_type.clone()));
            }
        }

        let mut merge_results: Vec<ValueId> = Vec::new();
        let merge_block = b.add_block(|merge| {
            for ret_type in return_types {
                merge_results.push(merge.add_parameter(ret_type.clone()));
            }
            merge.terminate_return(merge_results.clone());
        });

        let mut current_block = entry_block;

        if variants.len() == 1 {
            let variant_id = variants[0];
            let mut cb = b
                .block(current_block)
                .with_source_location(location.clone());
            let const_val = cb.u_const(32, variant_id.0 as u128);
            cb.assert_eq(fn_id_param, const_val);
            let call_results = cb.call(variant_id, forwarded_params.clone(), return_types.len());
            cb.terminate_jmp(merge_block, call_results);
        } else {
            for (i, &variant_id) in variants.iter().enumerate() {
                let is_last = i == variants.len() - 1;

                if is_last {
                    let mut cb = b
                        .block(current_block)
                        .with_source_location(location.clone());
                    let const_val = cb.u_const(32, variant_id.0 as u128);
                    cb.assert_eq(fn_id_param, const_val);
                    let call_results =
                        cb.call(variant_id, forwarded_params.clone(), return_types.len());
                    cb.terminate_jmp(merge_block, call_results);
                } else {
                    let call_block = b.add_block(|_| {});
                    let next_check_block = b.add_block(|_| {});

                    {
                        let mut cb = b
                            .block(current_block)
                            .with_source_location(location.clone());
                        let const_val = cb.u_const(32, variant_id.0 as u128);
                        let eq_result = cb.eq(fn_id_param, const_val);
                        cb.terminate_jmp_if(eq_result, call_block, next_check_block);
                    }

                    {
                        let mut cb = b.block(call_block).with_source_location(location.clone());
                        let call_results =
                            cb.call(variant_id, forwarded_params.clone(), return_types.len());
                        cb.terminate_jmp(merge_block, call_results);
                    }

                    current_block = next_check_block;
                }
            }
        }
    });

    dispatch_fn_id
}

/// Recursively replace `TypeExpr::Function` with `TypeExpr::U(32)` in a type.
fn replace_function_type(typ: &mut Type) {
    match &mut typ.expr {
        TypeExpr::Function => {
            typ.expr = TypeExpr::U(32);
        }
        TypeExpr::Array(inner, _) => replace_function_type(inner),
        TypeExpr::Slice(inner) => replace_function_type(inner),
        TypeExpr::Ref(inner) => replace_function_type(inner),
        TypeExpr::Tuple(elements) => {
            for elem in elements.iter_mut() {
                replace_function_type(elem);
            }
        }
        TypeExpr::Field
        | TypeExpr::U(_)
        | TypeExpr::I(_)
        | TypeExpr::WitnessOf(_)
        | TypeExpr::Blob(..) => {}
    }
}

/// Replace `TypeExpr::Function` in all type annotations within an instruction.
fn replace_function_types_in_instruction(instr: &mut OpCode) {
    match instr {
        OpCode::MkSeq { elem_type, .. } => replace_function_type(elem_type),
        OpCode::MkSeqOfBlob { element_type, .. } => replace_function_type(element_type),
        OpCode::MkRepeated { elem_type, .. } => replace_function_type(elem_type),
        OpCode::MkTuple { element_types, .. } => {
            for t in element_types.iter_mut() {
                replace_function_type(t);
            }
        }
        OpCode::FreshWitness { result_type, .. } => replace_function_type(result_type),
        OpCode::ReadGlobal { result_type, .. } => replace_function_type(result_type),
        OpCode::Todo { result_types, .. } => {
            for t in result_types.iter_mut() {
                replace_function_type(t);
            }
        }
        _ => {}
    }
}
