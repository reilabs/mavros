use std::collections::{HashMap, HashSet};

use crate::compiler::{
    ir::r#type::{Empty, Type, TypeExpr},
    pass_manager::{Pass, PassInfo, PassManager},
    ssa::{
        BlockId, CallTarget, Const, FunctionId, OpCode, Terminator, TupleIdx, ValueId, SSA,
    },
};

pub struct Defunctionalize {}

impl Defunctionalize {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass<Empty> for Defunctionalize {
    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "defunctionalize",
            needs: vec![],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        true
    }

    fn invalidates_types(&self) -> bool {
        true
    }

    fn run(&self, ssa: &mut SSA<Empty>, _pass_manager: &PassManager<Empty>) {
        run_defunctionalize(ssa);
    }
}

/// For each SSA value that may hold a function pointer, the set of
/// concrete FunctionIds it can point to.
type ReachingFns = HashMap<(FunctionId, ValueId), HashSet<FunctionId>>;

fn run_defunctionalize(ssa: &mut SSA<Empty>) {
    // Check if there are any FnPtrs at all
    let has_fn_ptrs = ssa.get_function_ids().collect::<Vec<_>>().iter().any(|fid| {
        ssa.get_function(*fid)
            .iter_consts()
            .any(|(_, c)| matches!(c, Const::FnPtr(_)))
    });
    if !has_fn_ptrs {
        return;
    }

    // Phase 1: Compute reaching definitions — which FnPtrs can reach each value
    let reaching = compute_reaching_fn_ptrs(ssa);

    // Phase 2: For each dynamic call site, build a dispatch function
    // with exactly the reachable targets
    let mut call_site_dispatch: HashMap<(FunctionId, ValueId), FunctionId> = HashMap::new();
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
            .unwrap_or_else(|| {
                panic!(
                    "No reaching FnPtrs for v{} in {:?}",
                    fn_ptr_val.0, fid
                )
            })
            .iter()
            .copied()
            .collect();

        assert!(!targets.is_empty(), "Empty target set for v{} in {:?}", fn_ptr_val.0, fid);

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

    // 3a. Replace Const::FnPtr → Const::U(32, ...) in all functions
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    for fid in &func_ids {
        let func = ssa.get_function(*fid);
        let to_replace: Vec<(ValueId, FunctionId)> = func
            .iter_consts()
            .filter_map(|(vid, c)| {
                if let Const::FnPtr(target) = c {
                    Some((*vid, *target))
                } else {
                    None
                }
            })
            .collect();

        let func = ssa.get_function_mut(*fid);
        for (vid, target) in to_replace {
            func.replace_const(vid, Const::U(32, target.0 as u128));
        }
    }

    // 3b. Replace CallTarget::Dynamic → CallTarget::Static(dispatch_fn)
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    for fid in func_ids {
        let func = ssa.get_function(fid);
        let has_dynamic = func.get_blocks().any(|(_, block)| {
            block.get_instructions().any(|instr| {
                matches!(instr, OpCode::Call { function: CallTarget::Dynamic(_), .. })
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
                match instr {
                    OpCode::Call {
                        results,
                        function: CallTarget::Dynamic(fn_ptr_val),
                        args,
                    } => {
                        let dispatch_fn = *call_site_dispatch
                            .get(&(fid, fn_ptr_val))
                            .expect(&format!(
                                "No dispatch function for v{} in {:?}",
                                fn_ptr_val.0, fid
                            ));
                        let mut new_args = Vec::with_capacity(args.len() + 1);
                        new_args.push(fn_ptr_val);
                        new_args.extend(args);

                        new_instructions.push(OpCode::Call {
                            results,
                            function: CallTarget::Static(dispatch_fn),
                            args: new_args,
                        });
                    }
                    other => new_instructions.push(other),
                }
            }

            let block = func.get_block_mut(bid);
            block.put_instructions(new_instructions);
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
                replace_function_types_in_instruction(instr);
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
fn compute_reaching_fn_ptrs(ssa: &SSA<Empty>) -> ReachingFns {
    let mut reaching: ReachingFns = HashMap::new();
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

    // Check if a type contains Function anywhere (for alias-aware propagation)
    fn contains_function(typ: &Type<Empty>) -> bool {
        match &typ.expr {
            TypeExpr::Function => true,
            TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) | TypeExpr::Ref(inner) => {
                contains_function(inner)
            }
            TypeExpr::Tuple(elems) => elems.iter().any(contains_function),
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::WitnessRef => false,
        }
    }

    // Pre-compute which values are Refs containing Functions (need bidirectional propagation)
    // These come from: Alloc results, block parameters with Ref<...Function...> type
    let mut is_ref_with_fn: HashSet<(FunctionId, ValueId)> = HashSet::new();
    for &fid in &func_ids {
        let func = ssa.get_function(fid);
        // Check Alloc instructions
        for (_bid, block) in func.get_blocks() {
            for instr in block.get_instructions() {
                if let OpCode::Alloc { result, elem_type, .. } = instr {
                    if contains_function(elem_type) {
                        is_ref_with_fn.insert((fid, *result));
                    }
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

    // Seed from FnPtr consts
    for &fid in &func_ids {
        let func = ssa.get_function(fid);
        for (vid, c) in func.iter_consts() {
            if let Const::FnPtr(target) = c {
                reaching
                    .entry((fid, *vid))
                    .or_default()
                    .insert(*target);
            }
        }
    }

    // Pre-compute return values per function
    let mut return_values: HashMap<FunctionId, Vec<ValueId>> = HashMap::new();
    for &fid in &func_ids {
        let func = ssa.get_function(fid);
        for (_bid, block) in func.get_blocks() {
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
    let mut global_slots: HashMap<usize, HashSet<FunctionId>> = HashMap::new();

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
                                changed |= propagate(&mut reaching, (fid, dest_params[i]), (fid, *arg));
                            }
                        }
                    }
                }

                for instr in block.get_instructions() {
                    match instr {
                        OpCode::Call { function, args, results } => {
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
                                    changed |= propagate(&mut reaching, (fid, *result), (fid, *elem));
                                }
                            }
                        }
                        OpCode::MkSeq { result, elems, .. } => {
                            for elem in elems {
                                changed |= propagate(&mut reaching, (fid, *elem), (fid, *result));
                                if is_ref_with_fn.contains(&(fid, *elem)) {
                                    changed |= propagate(&mut reaching, (fid, *result), (fid, *elem));
                                }
                            }
                        }
                        OpCode::ArrayGet { result, array, .. } => {
                            changed |= propagate(&mut reaching, (fid, *array), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *result)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *array));
                            }
                        }
                        OpCode::ArraySet { result, array, value, .. } => {
                            changed |= propagate(&mut reaching, (fid, *array), (fid, *result));
                            changed |= propagate(&mut reaching, (fid, *value), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *value)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *array));
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *value));
                            }
                        }
                        OpCode::TupleProj { result, tuple, .. } => {
                            changed |= propagate(&mut reaching, (fid, *tuple), (fid, *result));
                            if is_ref_with_fn.contains(&(fid, *result)) {
                                changed |= propagate(&mut reaching, (fid, *result), (fid, *tuple));
                            }
                        }
                        OpCode::Load { result, ptr } => {
                            changed |= propagate(&mut reaching, (fid, *ptr), (fid, *result));
                        }
                        OpCode::Store { ptr, value } => {
                            changed |= propagate(&mut reaching, (fid, *value), (fid, *ptr));
                        }
                        OpCode::SlicePush { result, slice, values, .. } => {
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
                        OpCode::Select { result, if_t, if_f, .. } => {
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
    ssa: &mut SSA<Empty>,
    counter: u32,
    param_types: &[Type<Empty>],
    return_types: &[Type<Empty>],
    variants: &[FunctionId],
) -> FunctionId {
    let dispatch_fn_id = ssa.add_function(format!("apply_dispatch@{}", counter));
    let func = ssa.get_function_mut(dispatch_fn_id);

    for ret_type in return_types {
        func.add_return_type(ret_type.clone());
    }

    let entry_block = func.get_entry_id();
    let fn_id_param = func.add_parameter(entry_block, Type::u32(Empty));

    let mut forwarded_params: Vec<ValueId> = Vec::new();
    for param_type in param_types {
        let p = func.add_parameter(entry_block, param_type.clone());
        forwarded_params.push(p);
    }

    let merge_block = func.add_block();
    let mut merge_results: Vec<ValueId> = Vec::new();
    for ret_type in return_types {
        let r = func.add_parameter(merge_block, ret_type.clone());
        merge_results.push(r);
    }
    func.terminate_block_with_return(merge_block, merge_results.clone());

    if variants.len() == 1 {
        let variant_id = variants[0];
        let const_val = func.push_u_const(32, variant_id.0 as u128);
        func.push_assert_eq(entry_block, fn_id_param, const_val);
        let call_results = func.push_call(
            entry_block,
            variant_id,
            forwarded_params.clone(),
            return_types.len(),
        );
        func.terminate_block_with_jmp(entry_block, merge_block, call_results);
    } else {
        let mut current_block = entry_block;

        for (i, &variant_id) in variants.iter().enumerate() {
            let is_last = i == variants.len() - 1;

            if is_last {
                let const_val = func.push_u_const(32, variant_id.0 as u128);
                func.push_assert_eq(current_block, fn_id_param, const_val);
                let call_results = func.push_call(
                    current_block,
                    variant_id,
                    forwarded_params.clone(),
                    return_types.len(),
                );
                func.terminate_block_with_jmp(current_block, merge_block, call_results);
            } else {
                let const_val = func.push_u_const(32, variant_id.0 as u128);
                let eq_result = func.push_eq(current_block, fn_id_param, const_val);

                let call_block = func.add_block();
                let next_check_block = func.add_block();

                func.terminate_block_with_jmp_if(
                    current_block,
                    eq_result,
                    call_block,
                    next_check_block,
                );

                let call_results = func.push_call(
                    call_block,
                    variant_id,
                    forwarded_params.clone(),
                    return_types.len(),
                );
                func.terminate_block_with_jmp(call_block, merge_block, call_results);

                current_block = next_check_block;
            }
        }
    }

    dispatch_fn_id
}

/// Recursively replace `TypeExpr::Function` with `TypeExpr::U(32)` in a type.
fn replace_function_type(typ: &mut Type<Empty>) {
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
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::WitnessRef => {}
    }
}

/// Replace `TypeExpr::Function` in all type annotations within an instruction.
fn replace_function_types_in_instruction(instr: &mut OpCode<Empty>) {
    match instr {
        OpCode::MkSeq { elem_type, .. } => replace_function_type(elem_type),
        OpCode::Alloc { elem_type, .. } => replace_function_type(elem_type),
        OpCode::MkTuple { element_types, .. } => {
            for t in element_types.iter_mut() {
                replace_function_type(t);
            }
        }
        OpCode::FreshWitness { result_type, .. } => replace_function_type(result_type),
        OpCode::ReadGlobal { result_type, .. } => replace_function_type(result_type),
        OpCode::TupleProj { idx, .. } => {
            if let TupleIdx::Dynamic(_, typ) = idx {
                replace_function_type(typ);
            }
        }
        OpCode::Todo { result_types, .. } => {
            for t in result_types.iter_mut() {
                replace_function_type(t);
            }
        }
        _ => {}
    }
}
