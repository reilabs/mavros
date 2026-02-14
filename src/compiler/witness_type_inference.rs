use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::ir::r#type::{Type, TypeExpr};
use crate::compiler::ssa::{
    BlockId, CallTarget, FunctionId, OpCode, SSA, SsaAnnotator, Terminator, TupleIdx, ValueId,
};
use crate::compiler::witness_info::{
    ConstantWitness, FunctionWitnessType, WitnessType,
};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Specialization key and value
// ---------------------------------------------------------------------------

#[derive(Eq, Hash, PartialEq, Clone, Debug)]
struct SpecKey {
    original_func_id: FunctionId,
    arg_types: Vec<WitnessType>,
    cfg_witness: ConstantWitness,
}

#[derive(Clone, Debug)]
struct SpecValue {
    specialized_func_id: FunctionId,
    return_types: Vec<WitnessType>,
    arg_types_out: Vec<WitnessType>,
}

// ---------------------------------------------------------------------------
// Per-function propagation result
// ---------------------------------------------------------------------------

struct PropagationResult {
    return_types: Vec<WitnessType>,
    arg_types_out: Vec<WitnessType>,
    value_wt: HashMap<ValueId, WitnessType>,
    block_cfg: HashMap<BlockId, ConstantWitness>,
    /// Call sites discovered: (callee_func_id, arg_types, cfg_witness, return_value_ids, arg_value_ids)
    call_sites: Vec<CallSiteInfo>,
}

#[derive(Clone)]
struct CallSiteInfo {
    callee_func_id: FunctionId,
    arg_types: Vec<WitnessType>,
    cfg_witness: ConstantWitness,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct WitnessTypeInference {
    functions: HashMap<FunctionId, FunctionWitnessType>,
}

impl WitnessTypeInference {
    pub fn new() -> Self {
        WitnessTypeInference {
            functions: HashMap::new(),
        }
    }

    pub fn get_function_witness_type(&self, func_id: FunctionId) -> &FunctionWitnessType {
        self.functions.get(&func_id).unwrap()
    }

    pub fn set_function_witness_type(
        &mut self,
        func_id: FunctionId,
        wt: FunctionWitnessType,
    ) {
        self.functions.insert(func_id, wt);
    }

    pub fn remove_function_witness_type(&mut self, func_id: FunctionId) {
        self.functions.remove(&func_id);
    }

    pub fn run(&mut self, ssa: &mut SSA, flow_analysis: &FlowAnalysis) -> Result<(), String> {
        let main_id = ssa.get_main_id();
        let main_func = ssa.get_function(main_id);

        // 1. Compute main's arg types: scalars=Witness, containers=Pure with Witness elems
        let main_arg_types: Vec<WitnessType> = main_func
            .get_entry()
            .get_parameters()
            .map(|(_, tp)| Self::main_arg_witness_type(tp))
            .collect();

        let main_return_types: Vec<WitnessType> = main_func
            .get_returns()
            .iter()
            .map(|_| WitnessType::Scalar(ConstantWitness::Pure)) // optimistic
            .collect();

        let main_cfg_witness = ConstantWitness::Pure;

        // 2. Register main specialization
        let main_key = SpecKey {
            original_func_id: main_id,
            arg_types: main_arg_types.clone(),
            cfg_witness: main_cfg_witness,
        };

        // Clone main function as specialized version, set as entry point
        let main_specialized = ssa.get_function(main_id).clone();
        let main_specialized_id = ssa.insert_function(main_specialized);
        ssa.set_entry_point(main_specialized_id);

        let mut specializations: HashMap<SpecKey, SpecValue> = HashMap::new();
        let mut worklist: VecDeque<SpecKey> = VecDeque::new();
        let mut queued: HashSet<SpecKey> = HashSet::new();
        // Track which specializations call which (for re-queuing callers)
        let mut callers: HashMap<SpecKey, HashSet<SpecKey>> = HashMap::new();

        specializations.insert(
            main_key.clone(),
            SpecValue {
                specialized_func_id: main_specialized_id,
                return_types: main_return_types,
                arg_types_out: main_arg_types.clone(),
            },
        );
        worklist.push_back(main_key.clone());
        queued.insert(main_key.clone());

        // 3. Global worklist loop
        while let Some(spec_key) = worklist.pop_front() {
            queued.remove(&spec_key);
            let func = ssa.get_function(spec_key.original_func_id);
            let cfg = flow_analysis.get_function_cfg(spec_key.original_func_id);

            let result = Self::propagate_function(
                func,
                cfg,
                &spec_key.arg_types,
                spec_key.cfg_witness,
                &specializations,
                ssa,
            );

            let spec_value = specializations.get_mut(&spec_key).unwrap();
            let changed = result.return_types != spec_value.return_types
                || result.arg_types_out != spec_value.arg_types_out;

            spec_value.return_types = result.return_types;
            spec_value.arg_types_out = result.arg_types_out;

            // Store the propagation result for later FunctionWitnessType construction
            let spec_func_id = spec_value.specialized_func_id;
            let fwt = Self::build_function_witness_type(
                &spec_key,
                spec_value,
                &result.value_wt,
                &result.block_cfg,
            );
            self.functions.insert(spec_func_id, fwt);

            // Process call sites: register new specializations
            for call_site in &result.call_sites {
                let callee_key = SpecKey {
                    original_func_id: call_site.callee_func_id,
                    arg_types: call_site.arg_types.clone(),
                    cfg_witness: call_site.cfg_witness,
                };

                // Track caller relationship
                callers
                    .entry(callee_key.clone())
                    .or_default()
                    .insert(spec_key.clone());

                if !specializations.contains_key(&callee_key) {
                    // New specialization: clone function, register with optimistic Pure returns
                    let callee_func = ssa.get_function(callee_key.original_func_id);
                    let callee_return_types: Vec<WitnessType> = callee_func
                        .get_returns()
                        .iter()
                        .map(|tp| Self::construct_pure_witness_for_type(tp))
                        .collect();

                    let specialized_clone = callee_func.clone();
                    let specialized_id = ssa.insert_function(specialized_clone);

                    specializations.insert(
                        callee_key.clone(),
                        SpecValue {
                            specialized_func_id: specialized_id,
                            return_types: callee_return_types,
                            arg_types_out: callee_key.arg_types.clone(),
                        },
                    );
                    queued.insert(callee_key.clone());
                    worklist.push_back(callee_key);
                }
            }

            // If output changed, re-queue callers
            if changed {
                if let Some(caller_set) = callers.get(&spec_key) {
                    for caller_key in caller_set {
                        if !queued.contains(caller_key) {
                            queued.insert(caller_key.clone());
                            worklist.push_back(caller_key.clone());
                        }
                    }
                }
            }
        }

        // 4. Update Call targets in specialized functions to point to specialized callees
        for (spec_key, spec_value) in &specializations {
            let func = ssa.get_function(spec_key.original_func_id);
            let cfg = flow_analysis.get_function_cfg(spec_key.original_func_id);

            // Re-propagate to get call sites with their correct mapping
            let result = Self::propagate_function(
                func,
                cfg,
                &spec_key.arg_types,
                spec_key.cfg_witness,
                &specializations,
                ssa,
            );

            let mut specialized_func = ssa.take_function(spec_value.specialized_func_id);

            // Build call site index: for each block, collect calls in order
            let block_queue: Vec<BlockId> = cfg.get_blocks_bfs().collect();
            let mut call_site_iter = result.call_sites.iter();

            for block_id in block_queue {
                let block = specialized_func.get_block_mut(block_id);
                for instruction in block.get_instructions_mut() {
                    if let OpCode::Call {
                        function: CallTarget::Static(func_id),
                        ..
                    } = instruction
                    {
                        let call_site = call_site_iter.next().unwrap();
                        let callee_key = SpecKey {
                            original_func_id: call_site.callee_func_id,
                            arg_types: call_site.arg_types.clone(),
                            cfg_witness: call_site.cfg_witness,
                        };
                        let callee_spec = specializations.get(&callee_key).unwrap();
                        *func_id = callee_spec.specialized_func_id;
                    }
                }
            }

            ssa.put_function(spec_value.specialized_func_id, specialized_func);

            // Store final FunctionWitnessType
            let final_fwt = Self::build_function_witness_type(
                spec_key,
                spec_value,
                &result.value_wt,
                &result.block_cfg,
            );
            self.functions.insert(spec_value.specialized_func_id, final_fwt);
        }

        // 5. Remove original unspecialized functions
        let original_func_ids: HashSet<FunctionId> = specializations
            .keys()
            .map(|k| k.original_func_id)
            .collect();
        let specialized_func_ids: HashSet<FunctionId> = specializations
            .values()
            .map(|v| v.specialized_func_id)
            .collect();
        for orig_id in original_func_ids {
            if !specialized_func_ids.contains(&orig_id) {
                ssa.take_function(orig_id);
            }
        }

        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Per-function forward propagation (inner fixed-point)
    // ---------------------------------------------------------------------------

    fn propagate_function(
        func: &crate::compiler::ssa::Function,
        cfg: &crate::compiler::flow_analysis::CFG,
        arg_types: &[WitnessType],
        cfg_witness: ConstantWitness,
        specializations: &HashMap<SpecKey, SpecValue>,
        ssa: &SSA,
    ) -> PropagationResult {
        let entry_id = func.get_entry_id();
        let block_queue: Vec<BlockId> = cfg.get_blocks_bfs().collect();

        // Inner state
        let mut value_wt: HashMap<ValueId, WitnessType> = HashMap::new();
        let mut block_cfg: HashMap<BlockId, ConstantWitness> = HashMap::new();
        let mut alloc_inner: HashMap<ValueId, WitnessType> = HashMap::new();

        // Initialize constants as Pure
        for (value_id, _) in func.iter_consts() {
            value_wt.insert(*value_id, WitnessType::Scalar(ConstantWitness::Pure));
        }

        // Initialize entry block params from arg_types
        let entry_params: Vec<(ValueId, Type)> =
            func.get_entry().get_parameters().cloned().collect();
        for ((value_id, _), wt) in entry_params.iter().zip(arg_types.iter()) {
            value_wt.insert(*value_id, wt.clone());
        }

        // Initialize other block params as Pure (optimistic)
        for (block_id, block) in func.get_blocks() {
            if *block_id == entry_id {
                block_cfg.insert(*block_id, cfg_witness);
                continue;
            }
            block_cfg.insert(*block_id, cfg_witness); // function minimum
            for (value_id, tp) in block.get_parameters() {
                if !value_wt.contains_key(value_id) {
                    value_wt.insert(*value_id, Self::construct_pure_witness_for_type(tp));
                }
            }
        }

        // Initialize alloc sites
        for block_id in &block_queue {
            let block = func.get_block(*block_id);
            for instruction in block.get_instructions() {
                if let OpCode::Alloc {
                    result,
                    elem_type: tp,
                } = instruction
                {
                    alloc_inner.insert(*result, Self::construct_pure_witness_for_type(tp));
                }
            }
        }

        // Inner iteration until stable
        let max_iterations = 100;
        for _iteration in 0..max_iterations {
            let old_value_wt = value_wt.clone();
            let old_block_cfg = block_cfg.clone();
            let old_alloc_inner = alloc_inner.clone();

            Self::propagate_once(
                func,
                cfg,
                &block_queue,
                entry_id,
                &mut value_wt,
                &mut block_cfg,
                &mut alloc_inner,
                specializations,
                ssa,
            );

            if value_wt == old_value_wt
                && block_cfg == old_block_cfg
                && alloc_inner == old_alloc_inner
            {
                break;
            }
        }

        // Collect return types
        let mut return_types = Vec::new();
        let mut call_sites = Vec::new();

        // Re-run one more time to collect call_sites and return_types
        // (they should be stable now)
        for block_id in &block_queue {
            let block = func.get_block(*block_id);
            let block_cw = *block_cfg.get(block_id).unwrap();

            for instruction in block.get_instructions() {
                if let OpCode::Call {
                    results: _,
                    function: CallTarget::Static(callee_id),
                    args,
                } = instruction
                {
                    let callee_arg_types: Vec<WitnessType> = args
                        .iter()
                        .map(|v| value_wt.get(v).unwrap().clone())
                        .collect();
                    call_sites.push(CallSiteInfo {
                        callee_func_id: *callee_id,
                        arg_types: callee_arg_types,
                        cfg_witness: block_cw,
                    });
                }
            }

            if let Some(Terminator::Return(values)) = block.get_terminator() {
                let ret_wts: Vec<WitnessType> = values
                    .iter()
                    .map(|v| value_wt.get(v).unwrap().clone())
                    .collect();
                if return_types.is_empty() {
                    return_types = ret_wts;
                } else {
                    return_types = return_types
                        .iter()
                        .zip(ret_wts.iter())
                        .map(|(a, b)| a.join(b))
                        .collect();
                }
            }
        }

        // If no return was found (shouldn't happen), default to pure
        if return_types.is_empty() {
            return_types = func
                .get_returns()
                .iter()
                .map(|tp| Self::construct_pure_witness_for_type(tp))
                .collect();
        }

        // Compute arg_types_out: for Ref args, reflect updated alloc_inner
        let arg_types_out: Vec<WitnessType> = entry_params
            .iter()
            .zip(arg_types.iter())
            .map(|((value_id, _), original_arg)| {
                match original_arg {
                    WitnessType::Ref(_, _) => {
                        // Look up the alloc_inner for this ref value
                        if let Some(inner) = alloc_inner.get(value_id) {
                            WitnessType::Ref(ConstantWitness::Pure, Box::new(inner.clone()))
                        } else {
                            original_arg.clone()
                        }
                    }
                    _ => original_arg.clone(),
                }
            })
            .collect();

        PropagationResult {
            return_types,
            arg_types_out,
            value_wt,
            block_cfg,
            call_sites,
        }
    }

    /// Single forward pass over all blocks
    fn propagate_once(
        func: &crate::compiler::ssa::Function,
        cfg: &crate::compiler::flow_analysis::CFG,
        block_queue: &[BlockId],
        _entry_id: BlockId,
        value_wt: &mut HashMap<ValueId, WitnessType>,
        block_cfg: &mut HashMap<BlockId, ConstantWitness>,
        alloc_inner: &mut HashMap<ValueId, WitnessType>,
        specializations: &HashMap<SpecKey, SpecValue>,
        ssa: &SSA,
    ) {
        for block_id in block_queue {
            let block = func.get_block(*block_id);
            let block_cw = *block_cfg.get(block_id).unwrap();

            for instruction in block.get_instructions() {
                match instruction {
                    OpCode::BinaryArithOp {
                        kind: _,
                        result: r,
                        lhs,
                        rhs,
                    }
                    | OpCode::Cmp {
                        kind: _,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let result_wt = Self::infer_binop_cmp_result(
                            value_wt.get(lhs).unwrap(),
                            value_wt.get(rhs).unwrap(),
                        );
                        value_wt.insert(*r, result_wt);
                    }
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t,
                        if_f,
                    } => {
                        let result_wt = Self::infer_select_result(
                            value_wt.get(cond).unwrap(),
                            value_wt.get(if_t).unwrap(),
                            value_wt.get(if_f).unwrap(),
                        );
                        value_wt.insert(*r, result_wt);
                    }
                    OpCode::Alloc {
                        result: r,
                        elem_type: _tp,
                    } => {
                        let inner = alloc_inner.get(r).unwrap().clone();
                        value_wt.insert(
                            *r,
                            WitnessType::Ref(ConstantWitness::Pure, Box::new(inner)),
                        );
                    }
                    OpCode::Store { ptr, value: v } => {
                        let val_wt = value_wt.get(v).unwrap();
                        // Find the alloc origin for this ptr
                        let origin = Self::find_alloc_origin(ptr, alloc_inner);
                        if let Some(origin_id) = origin {
                            let current_inner = alloc_inner.get(&origin_id).unwrap().clone();
                            // Join stored value into alloc inner;
                            // cfg_witness contributes to toplevel of what's stored
                            let store_wt = val_wt.with_toplevel_info(
                                val_wt.toplevel_info().join(block_cw),
                            );
                            // The alloc's elem_type may not match the actual stored value's
                            let new_inner = current_inner.join(&store_wt);
                            alloc_inner.insert(origin_id, new_inner);
                        }
                    }
                    OpCode::Load { result: r, ptr } => {
                        let ptr_wt = value_wt.get(ptr).unwrap();
                        let origin = Self::find_alloc_origin(ptr, alloc_inner);
                        if let Some(origin_id) = origin {
                            let inner = alloc_inner.get(&origin_id).unwrap().clone();
                            let ptr_toplevel = ptr_wt.toplevel_info();
                            let result_wt = inner.with_toplevel_info(
                                inner.toplevel_info().join(ptr_toplevel),
                            );
                            value_wt.insert(*r, result_wt);
                        } else {
                            // Ref param — use the Ref's inner type
                            match ptr_wt {
                                WitnessType::Ref(ptr_info, inner) => {
                                    value_wt.insert(
                                        *r,
                                        inner.with_toplevel_info(
                                            inner.toplevel_info().join(*ptr_info),
                                        ),
                                    );
                                }
                                _ => panic!("Load from non-ref type"),
                            }
                        }
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: _,
                        result_type: tp,
                    } => {
                        let result_wt = Self::construct_pure_witness_for_type(tp);
                        value_wt.insert(*r, result_wt);
                    }
                    OpCode::AssertEq { .. }
                    | OpCode::AssertR1C { .. }
                    | OpCode::InitGlobal { .. }
                    | OpCode::DropGlobal { .. }
                    | OpCode::Rangecheck { .. }
                    | OpCode::MemOp { .. } => {}
                    OpCode::ArrayGet {
                        result: r,
                        array: arr,
                        index: idx,
                    } => {
                        let arr_wt = value_wt.get(arr).unwrap();
                        let idx_wt = value_wt.get(idx).unwrap();
                        let elem_wt = arr_wt.child_witness_type().unwrap();
                        let result_wt = elem_wt.with_toplevel_info(
                            arr_wt
                                .toplevel_info()
                                .join(idx_wt.toplevel_info())
                                .join(elem_wt.toplevel_info()),
                        );
                        value_wt.insert(*r, result_wt);
                    }
                    OpCode::ArraySet {
                        result: r,
                        array: arr,
                        index: idx,
                        value,
                    } => {
                        let arr_wt = value_wt.get(arr).unwrap();
                        let idx_wt = value_wt.get(idx).unwrap();
                        let val_wt = value_wt.get(value).unwrap();
                        let arr_elem_wt = arr_wt.child_witness_type().unwrap();
                        // Join existing element with new value, taint by index witness-ness
                        let idx_info = idx_wt.toplevel_info();
                        let result_arr_elem = arr_elem_wt.join(val_wt).with_toplevel_info(
                            arr_elem_wt
                                .toplevel_info()
                                .join(val_wt.toplevel_info())
                                .join(idx_info),
                        );
                        let result_wt = WitnessType::Array(
                            arr_wt.toplevel_info(),
                            Box::new(result_arr_elem),
                        );
                        value_wt.insert(*r, result_wt);
                    }
                    OpCode::SlicePush {
                        dir: _,
                        result: r,
                        slice: sl,
                        values,
                    } => {
                        let slice_wt = value_wt.get(sl).unwrap();
                        let slice_elem_wt = slice_wt.child_witness_type().unwrap();
                        let mut result_elem_wt = slice_elem_wt.clone();
                        for value in values {
                            let val_wt = value_wt.get(value).unwrap();
                            result_elem_wt = result_elem_wt.join(val_wt);
                        }
                        let result_wt = WitnessType::Array(
                            slice_wt.toplevel_info(),
                            Box::new(result_elem_wt),
                        );
                        value_wt.insert(*r, result_wt);
                    }
                    OpCode::SliceLen { result: r, slice: _ } => {
                        value_wt.insert(*r, WitnessType::Scalar(ConstantWitness::Pure));
                    }
                    OpCode::Call {
                        results,
                        function: CallTarget::Static(callee_id),
                        args,
                    } => {
                        let callee_arg_types: Vec<WitnessType> = args
                            .iter()
                            .map(|v| value_wt.get(v).unwrap().clone())
                            .collect();
                        let callee_key = SpecKey {
                            original_func_id: *callee_id,
                            arg_types: callee_arg_types,
                            cfg_witness: block_cw,
                        };

                        if let Some(callee_spec) = specializations.get(&callee_key) {
                            // Use callee's return types
                            for (result, ret_wt) in
                                results.iter().zip(callee_spec.return_types.iter())
                            {
                                value_wt.insert(*result, ret_wt.clone());
                            }
                            // Update alloc_inner from arg_types_out for Ref args
                            for (arg, arg_out) in
                                args.iter().zip(callee_spec.arg_types_out.iter())
                            {
                                if let WitnessType::Ref(_, inner_out) = arg_out {
                                    let origin = Self::find_alloc_origin(arg, alloc_inner);
                                    if let Some(origin_id) = origin {
                                        let current_inner =
                                            alloc_inner.get(&origin_id).unwrap().clone();
                                        let new_inner = current_inner.join(inner_out);
                                        alloc_inner.insert(origin_id, new_inner);
                                    }
                                }
                            }
                        } else {
                            // Specialization not yet registered — use optimistic Pure returns
                            // based on the callee function's return type structure
                            let callee_func = ssa.get_function(*callee_id);
                            let callee_return_types = callee_func.get_returns();
                            for (result, ret_type) in
                                results.iter().zip(callee_return_types.iter())
                            {
                                let pure_wt = Self::construct_pure_witness_for_type(ret_type);
                                // Join with existing value if present (from prior iteration)
                                let result_wt = value_wt
                                    .get(result)
                                    .map(|existing| existing.join(&pure_wt))
                                    .unwrap_or(pure_wt);
                                value_wt.insert(*result, result_wt);
                            }
                        }
                    }
                    OpCode::Call {
                        function: CallTarget::Dynamic(_),
                        ..
                    } => {
                        panic!(
                            "Dynamic call targets are not supported in witness type inference"
                        );
                    }
                    OpCode::MkSeq {
                        result,
                        elems,
                        seq_type: _,
                        elem_type: tp,
                    } => {
                        let base = Self::construct_pure_witness_for_type(tp);
                        let result_wt = elems
                            .iter()
                            .fold(base, |acc, v| acc.join(value_wt.get(v).unwrap()));
                        value_wt.insert(
                            *result,
                            WitnessType::Array(ConstantWitness::Pure, Box::new(result_wt)),
                        );
                    }
                    OpCode::Cast {
                        result, value, ..
                    }
                    | OpCode::Truncate {
                        result,
                        value,
                        ..
                    }
                    | OpCode::Not { result, value } => {
                        let val_wt = value_wt.get(value).unwrap().clone();
                        value_wt.insert(*result, val_wt);
                    }
                    OpCode::ToBits {
                        result,
                        value,
                        ..
                    }
                    | OpCode::ToRadix {
                        result,
                        value,
                        ..
                    } => {
                        let val_wt = value_wt.get(value).unwrap().clone();
                        value_wt.insert(
                            *result,
                            WitnessType::Array(ConstantWitness::Pure, Box::new(val_wt)),
                        );
                    }
                    OpCode::TupleProj {
                        result,
                        tuple,
                        idx,
                    } => {
                        if let TupleIdx::Static(child_index) = idx {
                            let tuple_wt = value_wt.get(tuple).unwrap();
                            match tuple_wt {
                                WitnessType::Tuple(top, children) => {
                                    let elem_wt = &children[*child_index];
                                    let result_wt = elem_wt.with_toplevel_info(
                                        (*top).join(elem_wt.toplevel_info()),
                                    );
                                    value_wt.insert(*result, result_wt);
                                }
                                _ => {
                                    panic!(
                                        "TupleProj on non-tuple witness type: {:?}",
                                        tuple_wt
                                    );
                                }
                            }
                        } else {
                            panic!("Tuple index should be static at this stage");
                        }
                    }
                    OpCode::MkTuple {
                        result,
                        elems,
                        element_types: _,
                    } => {
                        let children: Vec<WitnessType> = elems
                            .iter()
                            .map(|v| value_wt.get(v).unwrap().clone())
                            .collect();
                        value_wt.insert(
                            *result,
                            WitnessType::Tuple(ConstantWitness::Pure, children),
                        );
                    }
                    OpCode::WriteWitness { .. }
                    | OpCode::Constrain { .. }
                    | OpCode::FreshWitness { .. }
                    | OpCode::BumpD { .. }
                    | OpCode::NextDCoeff { .. }
                    | OpCode::MulConst { .. }
                    | OpCode::Lookup { .. }
                    | OpCode::DLookup { .. }
                    | OpCode::Todo { .. }
                    | OpCode::ValueOf { .. } => {
                        panic!(
                            "Should not be present at this stage {:?}",
                            instruction
                        );
                    }
                }
            }

            // Handle terminator
            if let Some(terminator) = block.get_terminator() {
                match terminator {
                    Terminator::Return(_) => {
                        // Return types collected after convergence
                    }
                    Terminator::Jmp(target, params) => {
                        let target_params: Vec<(ValueId, Type)> = func
                            .get_block(*target)
                            .get_parameters()
                            .cloned()
                            .collect();
                        for ((target_value, _), param) in
                            target_params.iter().zip(params.iter())
                        {
                            let param_wt = value_wt.get(param).unwrap();
                            let existing = value_wt.get(target_value).unwrap();
                            let joined = existing.join(param_wt);
                            value_wt.insert(*target_value, joined);
                        }
                    }
                    Terminator::JmpIf(cond, _if_true, _if_false) => {
                        let cond_toplevel = value_wt.get(cond).unwrap().toplevel_info();
                        // Both loops and if-else: join cond into merge point params
                        // and body block cfg_witnesses
                        let merge = cfg.get_merge_point(*block_id);
                        let merge_params: Vec<ValueId> = func
                            .get_block(merge)
                            .get_parameters()
                            .map(|(v, _)| *v)
                            .collect();
                        for param_id in merge_params {
                            let existing = value_wt.get(&param_id).unwrap();
                            let joined = existing.with_toplevel_info(
                                existing.toplevel_info().join(cond_toplevel),
                            );
                            value_wt.insert(param_id, joined);
                        }

                        let body_blocks = cfg.get_if_body(*block_id);
                        for body_block_id in body_blocks {
                            let existing_cfg = *block_cfg.get(&body_block_id).unwrap();
                            let new_cfg = existing_cfg
                                .join(cond_toplevel)
                                .join(block_cw);
                            block_cfg.insert(body_block_id, new_cfg);
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Find the alloc origin for a pointer value.
    /// Returns Some(value_id) if the ptr corresponds to a local alloc site.
    /// Returns None for Ref-typed function parameters (external refs).
    fn find_alloc_origin(
        ptr: &ValueId,
        alloc_inner: &HashMap<ValueId, WitnessType>,
    ) -> Option<ValueId> {
        // In SSA, the alloc origin IS the ptr value itself if it was an Alloc instruction.
        // If there's no alloc_inner entry, it's an external ref (function parameter).
        if alloc_inner.contains_key(ptr) {
            Some(*ptr)
        } else {
            None
        }
    }

    fn build_function_witness_type(
        spec_key: &SpecKey,
        spec_value: &SpecValue,
        value_wt: &HashMap<ValueId, WitnessType>,
        block_cfg: &HashMap<BlockId, ConstantWitness>,
    ) -> FunctionWitnessType {
        FunctionWitnessType {
            returns_witness: spec_value.return_types.clone(),
            cfg_witness: spec_key.cfg_witness,
            parameters: spec_key.arg_types.clone(),
            block_cfg_witness: block_cfg.clone(),
            value_witness_types: value_wt.clone(),
        }
    }

    fn infer_binop_cmp_result(lhs_wt: &WitnessType, rhs_wt: &WitnessType) -> WitnessType {
        WitnessType::Scalar(lhs_wt.toplevel_info().join(rhs_wt.toplevel_info()))
    }

    fn infer_select_result(
        cond_wt: &WitnessType,
        then_wt: &WitnessType,
        otherwise_wt: &WitnessType,
    ) -> WitnessType {
        then_wt.join(otherwise_wt).with_toplevel_info(
            cond_wt
                .toplevel_info()
                .join(then_wt.toplevel_info())
                .join(otherwise_wt.toplevel_info()),
        )
    }

    /// Compute witness type for a main function argument.
    /// Scalars are Witness (private inputs), containers are Pure with Witness elements.
    fn main_arg_witness_type(tp: &Type) -> WitnessType {
        match &tp.expr {
            TypeExpr::U(_) | TypeExpr::Field => WitnessType::Scalar(ConstantWitness::Witness),
            TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
                WitnessType::Array(ConstantWitness::Pure, Box::new(Self::main_arg_witness_type(inner)))
            }
            TypeExpr::Tuple(elements) => WitnessType::Tuple(
                ConstantWitness::Pure,
                elements
                    .iter()
                    .map(|e| Self::main_arg_witness_type(e))
                    .collect(),
            ),
            TypeExpr::Ref(_) => panic!("Ref in main signature"),
            TypeExpr::WitnessOf(_) => panic!("WitnessOf should not be present at this stage"),
            TypeExpr::Function => WitnessType::Scalar(ConstantWitness::Pure),
        }
    }

    fn construct_pure_witness_for_type(typ: &Type) -> WitnessType {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::Field => WitnessType::Scalar(ConstantWitness::Pure),
            TypeExpr::Array(i, _) => WitnessType::Array(
                ConstantWitness::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Slice(i) => WitnessType::Array(
                ConstantWitness::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Ref(i) => WitnessType::Ref(
                ConstantWitness::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::WitnessOf(_) => {
                panic!("ICE: WitnessOf should not be present at this stage");
            }
            TypeExpr::Tuple(elements) => WitnessType::Tuple(
                ConstantWitness::Pure,
                elements
                    .iter()
                    .map(|e| Self::construct_pure_witness_for_type(e))
                    .collect(),
            ),
            TypeExpr::Function => WitnessType::Scalar(ConstantWitness::Pure),
        }
    }
}

impl SsaAnnotator for WitnessTypeInference {
    fn annotate_value(&self, function_id: FunctionId, value_id: ValueId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_value(function_id, value_id)
    }

    fn annotate_block(&self, function_id: FunctionId, block_id: BlockId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_block(function_id, block_id)
    }

    fn annotate_function(&self, function_id: FunctionId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_function(function_id)
    }
}
