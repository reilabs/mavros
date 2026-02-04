use std::collections::HashMap;

use crate::compiler::{
    ir::r#type::{Empty, Type, TypeExpr},
    pass_manager::{Pass, PassInfo, PassManager},
    ssa::{
        BlockId, CallTarget, Const, Function, FunctionId, OpCode, Terminator, TupleIdx, ValueId,
        SSA,
    },
};

/// A canonical call signature derived from function param/return types.
/// Uses Display representation for hashing since `Type<Empty>` doesn't implement `Hash`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CallSignature {
    param_keys: Vec<String>,
    return_keys: Vec<String>,
}

impl CallSignature {
    fn from_function(func: &Function<Empty>) -> Self {
        let param_keys = func
            .get_param_types()
            .iter()
            .map(|t| format!("{}", t))
            .collect();
        let return_keys = func
            .get_returns()
            .iter()
            .map(|t| format!("{}", t))
            .collect();
        CallSignature {
            param_keys,
            return_keys,
        }
    }
}

/// Stored alongside the signature key so we can build dispatch functions
/// with concrete types.
#[derive(Clone)]
struct SignatureInfo {
    param_types: Vec<Type<Empty>>,
    return_types: Vec<Type<Empty>>,
}

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

fn run_defunctionalize(ssa: &mut SSA<Empty>) {
    // Phase 1: Discovery — collect all FnPtr targets
    let mut fn_ptr_targets: Vec<FunctionId> = Vec::new();
    {
        let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        for fid in &func_ids {
            let func = ssa.get_function(*fid);
            for (_vid, c) in func.iter_consts() {
                if let Const::FnPtr(target) = c {
                    if !fn_ptr_targets.contains(target) {
                        fn_ptr_targets.push(*target);
                    }
                }
            }
        }
    }

    if fn_ptr_targets.is_empty() {
        return;
    }

    // Compute signatures and group targets by signature
    let mut sig_groups: HashMap<CallSignature, (SignatureInfo, Vec<FunctionId>)> = HashMap::new();
    for &target_fn_id in &fn_ptr_targets {
        let target_func = ssa.get_function(target_fn_id);
        let sig = CallSignature::from_function(target_func);
        let info = SignatureInfo {
            param_types: target_func.get_param_types(),
            return_types: target_func.get_returns().to_vec(),
        };
        sig_groups
            .entry(sig)
            .or_insert_with(|| (info, Vec::new()))
            .1
            .push(target_fn_id);
    }

    // Phase 2: Build dispatch functions
    let mut dispatch_map: HashMap<CallSignature, FunctionId> = HashMap::new();
    let mut dispatch_counter = 0u32;

    for (sig, (info, variants)) in &sig_groups {
        let dispatch_fn_id = build_dispatch_function(ssa, dispatch_counter, info, variants);
        dispatch_map.insert(sig.clone(), dispatch_fn_id);
        dispatch_counter += 1;
    }

    // Build value→dispatch mapping by tracing FnPtr consts through the SSA
    let ptr_to_dispatch = resolve_fn_ptr_dispatch(ssa, &dispatch_map);

    // Phase 3: Transformation

    // 3a. Replace Const::FnPtr → Const::U32 in all functions
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
        // Check if this function has any dynamic calls
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
                match instr {
                    OpCode::Call {
                        results,
                        function: CallTarget::Dynamic(fn_ptr_val),
                        args,
                    } => {
                        let dispatch_fn = *ptr_to_dispatch
                            .get(&(fid, fn_ptr_val))
                            .expect(&format!(
                                "No dispatch function resolved for v{} in function {:?}",
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

    // 3c. Replace TypeExpr::Function → TypeExpr::U(32) in type annotations
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    for fid in func_ids {
        let func = ssa.get_function_mut(fid);

        // Replace in return types
        let mut returns = func.take_returns();
        for ret_type in returns.iter_mut() {
            replace_function_type(ret_type);
        }
        for ret_type in returns {
            func.add_return_type(ret_type);
        }

        // Replace in block parameters and instructions
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

/// Resolve which dispatch function each dynamic call's fn_ptr value maps to.
///
/// Traces FnPtr consts through Jmp block-parameter edges (intra-function) and
/// static call argument edges (inter-function) using fixpoint iteration.
/// Returns a map: (FunctionId, ValueId) → dispatch FunctionId.
fn resolve_fn_ptr_dispatch(
    ssa: &SSA<Empty>,
    dispatch_map: &HashMap<CallSignature, FunctionId>,
) -> HashMap<(FunctionId, ValueId), FunctionId> {
    // Step 1: seed from FnPtr consts — map (func, value) → target FunctionId
    let mut value_to_target: HashMap<(FunctionId, ValueId), FunctionId> = HashMap::new();
    let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

    for &fid in &func_ids {
        let func = ssa.get_function(fid);
        for (vid, c) in func.iter_consts() {
            if let Const::FnPtr(target) = c {
                value_to_target.insert((fid, *vid), *target);
            }
        }
    }

    // Step 2: fixpoint — propagate through Jmp edges and static call args
    let mut changed = true;
    while changed {
        changed = false;
        for &fid in &func_ids {
            let func = ssa.get_function(fid);
            let entry_params: Vec<ValueId> = func
                .get_entry()
                .get_parameters()
                .map(|(vid, _)| *vid)
                .collect();

            for (_bid, block) in func.get_blocks() {
                // Propagate through Jmp edges (intra-function)
                if let Some(Terminator::Jmp(dest, args)) = block.get_terminator() {
                    let dest_params: Vec<ValueId> = func
                        .get_block(*dest)
                        .get_parameters()
                        .map(|(vid, _)| *vid)
                        .collect();
                    for (i, arg) in args.iter().enumerate() {
                        if i < dest_params.len() {
                            if let Some(&target) = value_to_target.get(&(fid, *arg)) {
                                if !value_to_target.contains_key(&(fid, dest_params[i])) {
                                    value_to_target.insert((fid, dest_params[i]), target);
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                // Propagate through static call args (inter-function)
                for instr in block.get_instructions() {
                    if let OpCode::Call {
                        function: CallTarget::Static(callee_id),
                        args,
                        ..
                    } = instr
                    {
                        let callee = ssa.get_function(*callee_id);
                        let callee_params: Vec<ValueId> = callee
                            .get_entry()
                            .get_parameters()
                            .map(|(vid, _)| *vid)
                            .collect();
                        for (i, arg) in args.iter().enumerate() {
                            if i >= callee_params.len() {
                                continue;
                            }
                            // Direct: arg is a known FnPtr
                            if let Some(&target) = value_to_target.get(&(fid, *arg)) {
                                if !value_to_target.contains_key(&(*callee_id, callee_params[i])) {
                                    value_to_target
                                        .insert((*callee_id, callee_params[i]), target);
                                    changed = true;
                                }
                            }
                            // Transitive: arg is an entry param already resolved
                            else if let Some(param_idx) =
                                entry_params.iter().position(|vid| vid == arg)
                            {
                                if let Some(&target) = value_to_target.get(&(fid, entry_params[param_idx])) {
                                    if !value_to_target
                                        .contains_key(&(*callee_id, callee_params[i]))
                                    {
                                        value_to_target
                                            .insert((*callee_id, callee_params[i]), target);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 3: map each (func, value) → dispatch function via the target's signature
    let mut result: HashMap<(FunctionId, ValueId), FunctionId> = HashMap::new();
    for ((fid, vid), target) in &value_to_target {
        let target_func = ssa.get_function(*target);
        let sig = CallSignature::from_function(target_func);
        if let Some(&dispatch_fn) = dispatch_map.get(&sig) {
            result.insert((*fid, *vid), dispatch_fn);
        }
    }
    result
}

/// Build a dispatch function for a signature group.
fn build_dispatch_function(
    ssa: &mut SSA<Empty>,
    counter: u32,
    info: &SignatureInfo,
    variants: &[FunctionId],
) -> FunctionId {
    let dispatch_fn_id = ssa.add_function(format!("apply_dispatch@{}", counter));
    let func = ssa.get_function_mut(dispatch_fn_id);

    for ret_type in &info.return_types {
        func.add_return_type(ret_type.clone());
    }

    let entry_block = func.get_entry_id();

    let fn_id_param = func.add_parameter(entry_block, Type::u32(Empty));

    let mut forwarded_params: Vec<ValueId> = Vec::new();
    for param_type in &info.param_types {
        let p = func.add_parameter(entry_block, param_type.clone());
        forwarded_params.push(p);
    }

    let merge_block = func.add_block();
    let mut merge_results: Vec<ValueId> = Vec::new();
    for ret_type in &info.return_types {
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
            info.return_types.len(),
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
                    info.return_types.len(),
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
                    info.return_types.len(),
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
        TypeExpr::Array(inner, _) => {
            replace_function_type(inner);
        }
        TypeExpr::Slice(inner) => {
            replace_function_type(inner);
        }
        TypeExpr::Ref(inner) => {
            replace_function_type(inner);
        }
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
        OpCode::MkSeq { elem_type, .. } => {
            replace_function_type(elem_type);
        }
        OpCode::Alloc { elem_type, .. } => {
            replace_function_type(elem_type);
        }
        OpCode::MkTuple { element_types, .. } => {
            for t in element_types.iter_mut() {
                replace_function_type(t);
            }
        }
        OpCode::FreshWitness { result_type, .. } => {
            replace_function_type(result_type);
        }
        OpCode::ReadGlobal { result_type, .. } => {
            replace_function_type(result_type);
        }
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
