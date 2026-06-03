//! Performs whole program analysis to determine which values are potentially witness tainted, which
//! are _only_ witnesses, and which are only non-witness values.

use std::collections::{HashMap, HashSet, VecDeque};

use super::witness_info::{FunctionWitnessType, WitnessShape, WitnessType};
use crate::compiler::{
    analysis::flow_analysis::{self, FlowAnalysis},
    ssa::{
        BlockId, FunctionId, SSAAnotator, Terminator, ValueId,
        hlssa::{CallTarget, Constant, HLBlock, HLFunction, HLSSA, OpCode, Type, TypeExpr},
    },
};

// ---------------------------------------------------------------------------
// Specialization key and value
// ---------------------------------------------------------------------------

#[derive(Eq, Hash, PartialEq, Clone, Debug)]
struct SpecKey {
    original_func_id: FunctionId,
    arg_types: Vec<WitnessShape>,
    arg_aliases: Vec<Vec<ParamPtrPath>>,
    cfg_witness: WitnessType,
}

#[derive(Clone, Debug)]
struct SpecValue {
    specialized_func_id: FunctionId,
    return_types: Vec<WitnessShape>,
    return_aliases: Vec<HashSet<ParamPtrPath>>,
    arg_types_out: Vec<WitnessShape>,
}

// ---------------------------------------------------------------------------
// Per-function propagation result
// ---------------------------------------------------------------------------

struct PropagationResult {
    return_types: Vec<WitnessShape>,
    return_aliases: Vec<HashSet<ParamPtrPath>>,
    arg_types_out: Vec<WitnessShape>,
    value_wt: HashMap<ValueId, WitnessShape>,
    block_cfg: HashMap<BlockId, WitnessType>,
    /// Call sites discovered: (callee_func_id, arg_types, cfg_witness, return_value_ids, arg_value_ids)
    call_sites: Vec<CallSiteInfo>,
}

#[derive(Clone)]
struct CallSiteInfo {
    callee_func_id: FunctionId,
    arg_types: Vec<WitnessShape>,
    arg_aliases: Vec<Vec<ParamPtrPath>>,
    cfg_witness: WitnessType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PtrPath {
    root: ValueId,
    fields: Vec<usize>,
}

impl PtrPath {
    fn root(root: ValueId) -> Self {
        Self {
            root,
            fields: Vec::new(),
        }
    }

    fn field(&self, field_idx: usize) -> Self {
        let mut fields = self.fields.clone();
        fields.push(field_idx);
        Self {
            root: self.root,
            fields,
        }
    }

    fn extend(&self, fields: &[usize]) -> Self {
        let mut extended = self.fields.clone();
        extended.extend(fields.iter().copied());
        Self {
            root: self.root,
            fields: extended,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ParamPtrPath {
    param_idx: usize,
    fields: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct WitnessTypeInference {
    functions: HashMap<FunctionId, FunctionWitnessType>,
}

impl WitnessTypeInference {
    pub fn new() -> Self {
        WitnessTypeInference {
            functions: HashMap::new(),
        }
    }

    pub fn try_get_function_witness_type(
        &self,
        func_id: FunctionId,
    ) -> Option<&FunctionWitnessType> {
        self.functions.get(&func_id)
    }

    pub fn set_function_witness_type(&mut self, func_id: FunctionId, wt: FunctionWitnessType) {
        self.functions.insert(func_id, wt);
    }

    pub fn remove_function_witness_type(&mut self, func_id: FunctionId) {
        self.functions.remove(&func_id);
    }

    pub fn run(&mut self, ssa: &mut HLSSA, flow_analysis: &FlowAnalysis) -> Result<(), String> {
        let main_id = ssa.get_main_id();
        let main_func = ssa.get_function(main_id);

        // 1. Compute main's arg types: all Pure (WriteWitness in PrepareEntryPoint bridges to Witness)
        let main_arg_types: Vec<WitnessShape> = main_func
            .get_entry()
            .get_parameters()
            .map(|(_, tp)| Self::construct_pure_witness_for_type(tp))
            .collect();
        let main_arg_aliases = Self::identity_param_aliases(&main_arg_types);

        let main_return_types: Vec<WitnessShape> = main_func
            .get_returns()
            .iter()
            .map(|_| WitnessShape::Scalar(WitnessType::Pure)) // optimistic
            .collect();
        let main_return_len = main_func.get_returns().len();

        let main_cfg_witness = WitnessType::Pure;

        // 2. Register main specialization
        let main_key = SpecKey {
            original_func_id: main_id,
            arg_types: main_arg_types.clone(),
            arg_aliases: main_arg_aliases.clone(),
            cfg_witness: main_cfg_witness,
        };

        // Clone main function as specialized version, set as entry point
        let main_specialized_id = ssa.duplicate_function(main_id);
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
                return_aliases: vec![HashSet::new(); main_return_len],
                arg_types_out: main_arg_types.clone(),
            },
        );
        worklist.push_back(main_key.clone());
        queued.insert(main_key.clone());

        // 3. Global worklist loop
        while let Some(spec_key) = worklist.pop_front() {
            queued.remove(&spec_key);
            let specialized_func_id = specializations.get(&spec_key).unwrap().specialized_func_id;
            let func = ssa.get_function(specialized_func_id);
            let cfg = flow_analysis.get_function_cfg(spec_key.original_func_id);

            let result = Self::propagate_function(
                func,
                cfg,
                &spec_key.arg_types,
                &spec_key.arg_aliases,
                spec_key.cfg_witness,
                &specializations,
                ssa,
            );

            let spec_value = specializations.get_mut(&spec_key).unwrap();
            let changed = result.return_types != spec_value.return_types
                || result.return_aliases != spec_value.return_aliases
                || result.arg_types_out != spec_value.arg_types_out;

            spec_value.return_types = result.return_types;
            spec_value.return_aliases = result.return_aliases;
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
                    arg_aliases: call_site.arg_aliases.clone(),
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
                    let callee_return_types: Vec<WitnessShape> = callee_func
                        .get_returns()
                        .iter()
                        .map(Self::construct_pure_witness_for_type)
                        .collect();
                    let callee_return_len = callee_func.get_returns().len();

                    let specialized_id = ssa.duplicate_function(callee_key.original_func_id);

                    specializations.insert(
                        callee_key.clone(),
                        SpecValue {
                            specialized_func_id: specialized_id,
                            return_types: callee_return_types,
                            return_aliases: vec![HashSet::new(); callee_return_len],
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
            let func = ssa.get_function(spec_value.specialized_func_id);
            let cfg = flow_analysis.get_function_cfg(spec_key.original_func_id);

            // Re-propagate to get call sites with their correct mapping
            let result = Self::propagate_function(
                func,
                cfg,
                &spec_key.arg_types,
                &spec_key.arg_aliases,
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
                        unconstrained,
                        ..
                    } = instruction
                    {
                        if *unconstrained {
                            // Skip unconstrained calls — not specialized
                            continue;
                        }
                        let call_site = call_site_iter.next().unwrap();
                        let callee_key = SpecKey {
                            original_func_id: call_site.callee_func_id,
                            arg_types: call_site.arg_types.clone(),
                            arg_aliases: call_site.arg_aliases.clone(),
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
            self.functions
                .insert(spec_value.specialized_func_id, final_fwt);
        }

        // Original functions are now unreachable (entry point was moved to
        // specialized main). RemoveUnreachableFunctions cleans them up.

        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Per-function forward propagation (inner fixed-point)
    // ---------------------------------------------------------------------------

    fn propagate_function(
        func: &HLFunction,
        cfg: &flow_analysis::CFG,
        arg_types: &[WitnessShape],
        arg_aliases: &[Vec<ParamPtrPath>],
        cfg_witness: WitnessType,
        specializations: &HashMap<SpecKey, SpecValue>,
        ssa: &HLSSA,
    ) -> PropagationResult {
        let entry_id = func.get_entry_id();
        let block_queue: Vec<BlockId> = cfg.get_blocks_bfs().collect();

        // Inner state
        let mut value_wt: HashMap<ValueId, WitnessShape> = HashMap::new();
        let mut block_cfg: HashMap<BlockId, WitnessType> = HashMap::new();
        let mut ptr_aliases: HashMap<ValueId, HashSet<PtrPath>> = HashMap::new();
        let mut memory_wt: HashMap<ValueId, WitnessShape> = HashMap::new();

        ssa.for_each_const(|vid, val| {
            let shape = match val.as_ref() {
                Constant::U(_, _) | Constant::I(_, _) | Constant::Field(_) | Constant::FnPtr(_) => {
                    WitnessShape::Scalar(WitnessType::Pure)
                }
            };
            value_wt.insert(*vid, shape);
        });

        // Initialize entry block params from arg_types
        let entry_params: Vec<(ValueId, Type)> =
            func.get_entry().get_parameters().cloned().collect();
        let entry_param_indices = entry_params
            .iter()
            .enumerate()
            .map(|(idx, (value_id, _))| (*value_id, idx))
            .collect::<HashMap<_, _>>();
        for ((value_id, _), wt) in entry_params.iter().zip(arg_types.iter()) {
            value_wt.insert(*value_id, wt.clone());
            Self::init_ref_location(&mut ptr_aliases, &mut memory_wt, *value_id, wt);
        }
        Self::init_entry_ref_aliases(&mut ptr_aliases, &entry_params, arg_aliases);

        // Initialize other block params as Pure (optimistic)
        for (block_id, block) in func.get_blocks() {
            if *block_id == entry_id {
                block_cfg.insert(*block_id, cfg_witness);
                continue;
            }
            block_cfg.insert(*block_id, cfg_witness); // function minimum
            for (value_id, tp) in block.get_parameters() {
                if !value_wt.contains_key(value_id) {
                    let wt = Self::construct_pure_witness_for_type(tp);
                    value_wt.insert(*value_id, wt.clone());
                    Self::init_ref_location(&mut ptr_aliases, &mut memory_wt, *value_id, &wt);
                }
            }
        }

        // Inner iteration until stable
        let max_iterations = 100;
        for _iteration in 0..max_iterations {
            let old_value_wt = value_wt.clone();
            let old_block_cfg = block_cfg.clone();
            let old_ptr_aliases = ptr_aliases.clone();
            let old_memory_wt = memory_wt.clone();

            Self::propagate_once(
                func,
                cfg,
                &block_queue,
                entry_id,
                &mut value_wt,
                &mut block_cfg,
                &mut ptr_aliases,
                &mut memory_wt,
                specializations,
                ssa,
            );

            if value_wt == old_value_wt
                && block_cfg == old_block_cfg
                && ptr_aliases == old_ptr_aliases
                && memory_wt == old_memory_wt
            {
                break;
            }
        }

        // Collect return types
        let mut return_types = Vec::new();
        let mut return_aliases: Option<Vec<HashSet<ParamPtrPath>>> = None;
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
                    unconstrained,
                } = instruction
                {
                    if *unconstrained {
                        // Skip unconstrained calls — not specialized
                        continue;
                    }
                    let callee_arg_types: Vec<WitnessShape> = args
                        .iter()
                        .map(|v| value_wt.get(v).unwrap().clone())
                        .collect();
                    let callee_arg_aliases = Self::call_arg_aliases(args, &ptr_aliases);
                    call_sites.push(CallSiteInfo {
                        callee_func_id: *callee_id,
                        arg_types: callee_arg_types,
                        arg_aliases: callee_arg_aliases,
                        cfg_witness: block_cw,
                    });
                }
            }

            if let Some(Terminator::Return(values)) = block.get_terminator() {
                let ret_wts: Vec<WitnessShape> = values
                    .iter()
                    .map(|v| value_wt.get(v).unwrap().clone())
                    .collect();
                let ret_aliases = values
                    .iter()
                    .map(|v| Self::return_param_aliases(*v, &ptr_aliases, &entry_param_indices))
                    .collect::<Vec<_>>();
                if return_types.is_empty() {
                    return_types = ret_wts;
                    return_aliases = Some(ret_aliases);
                } else {
                    return_types = return_types
                        .iter()
                        .zip(ret_wts.iter())
                        .map(|(a, b)| a.join(b))
                        .collect();
                    let aliases = return_aliases.as_mut().unwrap();
                    for (existing, new_aliases) in aliases.iter_mut().zip(ret_aliases.into_iter()) {
                        existing.extend(new_aliases);
                    }
                }
            }
        }

        // If no return was found (shouldn't happen), default to pure
        if return_types.is_empty() {
            return_types = func
                .get_returns()
                .iter()
                .map(Self::construct_pure_witness_for_type)
                .collect();
        }
        let return_aliases =
            return_aliases.unwrap_or_else(|| vec![HashSet::new(); return_types.len()]);

        // Compute arg_types_out: for Ref args, reflect any updates to the ref's inner shape.
        let arg_types_out: Vec<WitnessShape> = entry_params
            .iter()
            .zip(arg_types.iter())
            .map(|((value_id, _), original_arg)| match original_arg {
                WitnessShape::Ref(_, _) => value_wt
                    .get(value_id)
                    .cloned()
                    .unwrap_or_else(|| original_arg.clone()),
                _ => original_arg.clone(),
            })
            .collect();

        PropagationResult {
            return_types,
            return_aliases,
            arg_types_out,
            value_wt,
            block_cfg,
            call_sites,
        }
    }

    /// Single forward pass over all blocks
    fn propagate_once(
        func: &HLFunction,
        cfg: &flow_analysis::CFG,
        block_queue: &[BlockId],
        _entry_id: BlockId,
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        block_cfg: &mut HashMap<BlockId, WitnessType>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        specializations: &HashMap<SpecKey, SpecValue>,
        ssa: &HLSSA,
    ) {
        for block_id in block_queue {
            let block = func.get_block(*block_id);
            let block_cw = *block_cfg.get(block_id).unwrap();

            Self::propagate_block_instructions(
                block,
                block_cw,
                value_wt,
                ptr_aliases,
                memory_wt,
                specializations,
                ssa,
            );

            // Handle terminator
            if let Some(terminator) = block.get_terminator() {
                match terminator {
                    Terminator::Return(_) => {
                        // Return types collected after convergence
                    }
                    Terminator::Jmp(target, params) => {
                        let target_params: Vec<(ValueId, Type)> =
                            func.get_block(*target).get_parameters().cloned().collect();
                        for ((target_value, _), param) in target_params.iter().zip(params.iter()) {
                            let param_wt = value_wt.get(param).unwrap().clone();
                            let existing = value_wt.get(target_value).unwrap().clone();
                            let joined = existing.join(&param_wt);
                            value_wt.insert(*target_value, joined.clone());
                            Self::merge_value_wt(value_wt, *param, joined);
                            if matches!(&param_wt, WitnessShape::Ref(_, _)) {
                                Self::join_ref_aliases(
                                    value_wt,
                                    ptr_aliases,
                                    memory_wt,
                                    *target_value,
                                    *param,
                                );
                            }
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
                            let joined = existing.with_witness_in_leaves(cond_toplevel);
                            value_wt.insert(param_id, joined);
                        }

                        let body_blocks = cfg.get_if_body(*block_id);
                        for body_block_id in body_blocks {
                            let existing_cfg = *block_cfg.get(&body_block_id).unwrap();
                            let new_cfg = existing_cfg.join(cond_toplevel).join(block_cw);
                            block_cfg.insert(body_block_id, new_cfg);
                        }
                    }
                }
            }
        }

        Self::sync_all_ref_values(value_wt, ptr_aliases, memory_wt);
    }

    /// Process all instructions in a single block, updating `value_wt` according
    /// to each opcode's witness-type rules.
    fn propagate_block_instructions(
        block: &HLBlock,
        block_cw: WitnessType,
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        specializations: &HashMap<SpecKey, SpecValue>,
        ssa: &HLSSA,
    ) {
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
                    elem_type: tp,
                } => {
                    let inner = Self::construct_pure_witness_for_type(tp);
                    let wt = WitnessShape::Ref(WitnessType::Pure, Box::new(inner));
                    value_wt.insert(*r, wt.clone());
                    Self::init_ref_location(ptr_aliases, memory_wt, *r, &wt);
                }
                OpCode::Store { ptr, value: v } => {
                    let val_wt = value_wt.get(v).unwrap();
                    // cfg_witness contributes to the toplevel of what's stored.
                    let store_wt = val_wt.with_toplevel_info(val_wt.toplevel_info().join(block_cw));
                    Self::join_ref_inner(value_wt, ptr_aliases, memory_wt, *ptr, &store_wt);
                }
                OpCode::Load { result: r, ptr } => {
                    let result_wt = Self::read_ref_inner(value_wt, ptr_aliases, memory_wt, *ptr);
                    value_wt.insert(*r, result_wt);
                }
                OpCode::RefTupleSplice {
                    result: r,
                    tuple_ref,
                    field_idx,
                } => {
                    Self::propagate_ref_tuple_splice(
                        value_wt,
                        ptr_aliases,
                        memory_wt,
                        *tuple_ref,
                        *r,
                        *field_idx,
                    );
                }
                OpCode::ReadGlobal {
                    result: r,
                    offset: _,
                    result_type: tp,
                } => {
                    let result_wt = Self::construct_pure_witness_for_type(tp);
                    value_wt.insert(*r, result_wt);
                    Self::init_ref_location(ptr_aliases, memory_wt, *r, value_wt.get(r).unwrap());
                }
                OpCode::Assert { .. }
                | OpCode::AssertCmp { .. }
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
                    let pushed_info = arr_wt.toplevel_info().join(idx_wt.toplevel_info());
                    let result_wt = elem_wt.with_witness_in_leaves(pushed_info);
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
                    let result_wt =
                        WitnessShape::Array(arr_wt.toplevel_info(), Box::new(result_arr_elem));
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
                    let result_wt =
                        WitnessShape::Array(slice_wt.toplevel_info(), Box::new(result_elem_wt));
                    value_wt.insert(*r, result_wt);
                }
                OpCode::SliceLen {
                    result: r,
                    slice: _,
                } => {
                    value_wt.insert(*r, WitnessShape::Scalar(WitnessType::Pure));
                }
                OpCode::Call {
                    results,
                    function: CallTarget::Static(callee_id),
                    args,
                    unconstrained,
                } => {
                    if *unconstrained {
                        // Unconstrained calls produce pure witness values
                        // (their outputs go through WriteWitness in PrepareEntryPoint)
                        let callee_func = ssa.get_function(*callee_id);
                        let callee_return_types = callee_func.get_returns();
                        for (result, ret_type) in results.iter().zip(callee_return_types.iter()) {
                            let pure_wt = Self::construct_pure_witness_for_type(ret_type);
                            value_wt.insert(*result, pure_wt);
                            Self::init_ref_location(
                                ptr_aliases,
                                memory_wt,
                                *result,
                                value_wt.get(result).unwrap(),
                            );
                        }
                    } else {
                        let callee_arg_types: Vec<WitnessShape> = args
                            .iter()
                            .map(|v| value_wt.get(v).unwrap().clone())
                            .collect();
                        let callee_arg_aliases = Self::call_arg_aliases(args, ptr_aliases);
                        let callee_key = SpecKey {
                            original_func_id: *callee_id,
                            arg_types: callee_arg_types,
                            arg_aliases: callee_arg_aliases,
                            cfg_witness: block_cw,
                        };

                        if let Some(callee_spec) = specializations.get(&callee_key) {
                            // Use callee's return types
                            for (result_idx, (result, ret_wt)) in results
                                .iter()
                                .zip(callee_spec.return_types.iter())
                                .enumerate()
                            {
                                value_wt.insert(*result, ret_wt.clone());
                                Self::init_ref_location(ptr_aliases, memory_wt, *result, ret_wt);
                                if matches!(ret_wt, WitnessShape::Ref(_, _)) {
                                    Self::apply_return_aliases(
                                        value_wt,
                                        ptr_aliases,
                                        memory_wt,
                                        *result,
                                        &callee_spec.return_aliases[result_idx],
                                        args,
                                    );
                                }
                            }
                            // Join callee ref-arg effects back into caller argument shapes.
                            for (arg, arg_out) in args.iter().zip(callee_spec.arg_types_out.iter())
                            {
                                Self::merge_value_wt(value_wt, *arg, arg_out.clone());
                                if let WitnessShape::Ref(_, inner_out) = arg_out {
                                    Self::join_ref_inner(
                                        value_wt,
                                        ptr_aliases,
                                        memory_wt,
                                        *arg,
                                        inner_out,
                                    );
                                }
                            }
                            Self::sync_all_ref_values(value_wt, ptr_aliases, memory_wt);
                        } else {
                            // Specialization not yet registered — use optimistic Pure returns
                            // based on the callee function's return type structure
                            let callee_func = ssa.get_function(*callee_id);
                            let callee_return_types = callee_func.get_returns();
                            for (result, ret_type) in results.iter().zip(callee_return_types.iter())
                            {
                                let pure_wt = Self::construct_pure_witness_for_type(ret_type);
                                // Join with existing value if present (from prior iteration)
                                let result_wt = value_wt
                                    .get(result)
                                    .map(|existing| existing.join(&pure_wt))
                                    .unwrap_or(pure_wt);
                                value_wt.insert(*result, result_wt);
                                Self::init_ref_location(
                                    ptr_aliases,
                                    memory_wt,
                                    *result,
                                    value_wt.get(result).unwrap(),
                                );
                            }
                        }
                    }
                }
                OpCode::Call {
                    function: CallTarget::Dynamic(_),
                    ..
                } => {
                    panic!("Dynamic call targets are not supported in witness type inference");
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
                        WitnessShape::Array(WitnessType::Pure, Box::new(result_wt)),
                    );
                }
                OpCode::MkRepeated {
                    result,
                    element,
                    seq_type: _,
                    count: _,
                    elem_type: tp,
                } => {
                    let base = Self::construct_pure_witness_for_type(tp);
                    let result_wt = base.join(value_wt.get(element).unwrap());
                    value_wt.insert(
                        *result,
                        WitnessShape::Array(WitnessType::Pure, Box::new(result_wt)),
                    );
                }
                OpCode::Spread { result, value, .. } => {
                    let val_wt = value_wt.get(value).unwrap().clone();
                    value_wt.insert(*result, val_wt);
                }
                OpCode::Unspread {
                    result_odd,
                    result_even,
                    value,
                    ..
                } => {
                    let val_wt = value_wt.get(value).unwrap().clone();
                    value_wt.insert(*result_odd, val_wt.clone());
                    value_wt.insert(*result_even, val_wt);
                }
                OpCode::Cast { result, value, .. }
                | OpCode::SExt { result, value, .. }
                | OpCode::BitRange { result, value, .. }
                | OpCode::Not { result, value } => {
                    let val_wt = value_wt.get(value).unwrap().clone();
                    value_wt.insert(*result, val_wt);
                }
                OpCode::ToBits { result, value, .. } | OpCode::ToRadix { result, value, .. } => {
                    let val_wt = value_wt.get(value).unwrap().clone();
                    value_wt.insert(
                        *result,
                        WitnessShape::Array(WitnessType::Pure, Box::new(val_wt)),
                    );
                }
                OpCode::TupleProj { result, tuple, idx } => {
                    let tuple_wt = value_wt.get(tuple).unwrap().clone();
                    match &tuple_wt {
                        WitnessShape::Tuple(top, children) => {
                            let elem_wt = &children[*idx];
                            let result_wt =
                                elem_wt.with_toplevel_info((*top).join(elem_wt.toplevel_info()));
                            Self::merge_value_wt(value_wt, *result, result_wt);
                            Self::join_tuple_child(
                                value_wt,
                                *tuple,
                                *idx,
                                value_wt.get(result).unwrap().clone(),
                            );
                        }
                        _ => {
                            panic!("TupleProj on non-tuple witness type: {:?}", tuple_wt);
                        }
                    }
                }
                OpCode::MkTuple { result, elems, .. } => {
                    let children: Vec<WitnessShape> = elems
                        .iter()
                        .map(|v| value_wt.get(v).unwrap().clone())
                        .collect();
                    Self::merge_value_wt(
                        value_wt,
                        *result,
                        WitnessShape::Tuple(WitnessType::Pure, children),
                    );
                    if let WitnessShape::Tuple(_, children) = value_wt.get(result).unwrap().clone()
                    {
                        for (elem, child) in elems.iter().zip(children.into_iter()) {
                            Self::merge_value_wt(value_wt, *elem, child);
                        }
                    }
                }
                OpCode::WriteWitness { result, .. } => {
                    // WriteWitness records a value on the witness tape.
                    // Its output is always Witness-typed.
                    if let Some(result) = result {
                        value_wt.insert(*result, WitnessShape::Scalar(WitnessType::Witness));
                    }
                }
                OpCode::Constrain { .. }
                | OpCode::FreshWitness { .. }
                | OpCode::BumpD { .. }
                | OpCode::NextDCoeff { .. }
                | OpCode::MulConst { .. }
                | OpCode::Lookup { .. }
                | OpCode::DLookup { .. }
                | OpCode::Todo { .. }
                | OpCode::ValueOf { .. } => {
                    panic!("Should not be present at this stage {:?}", instruction);
                }
                OpCode::Guard { .. } => {
                    panic!("ICE: Guard should not be present during witness type inference");
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn merge_value_wt(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        value: ValueId,
        new_wt: WitnessShape,
    ) {
        let joined = value_wt
            .get(&value)
            .map(|existing| existing.join(&new_wt))
            .unwrap_or(new_wt);
        value_wt.insert(value, joined);
    }

    fn join_tuple_child(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        tuple: ValueId,
        idx: usize,
        child_wt: WitnessShape,
    ) {
        let tuple_wt = value_wt
            .get(&tuple)
            .unwrap_or_else(|| panic!("Missing tuple witness type for {:?}", tuple))
            .clone();
        let updated = match tuple_wt {
            WitnessShape::Tuple(top, mut children) => {
                children[idx] = children[idx].join(&child_wt);
                WitnessShape::Tuple(top, children)
            }
            other => panic!("Tuple child update on non-tuple witness type: {:?}", other),
        };
        Self::merge_value_wt(value_wt, tuple, updated);
    }

    fn identity_param_aliases(arg_types: &[WitnessShape]) -> Vec<Vec<ParamPtrPath>> {
        arg_types
            .iter()
            .enumerate()
            .map(|(param_idx, wt)| {
                if matches!(wt, WitnessShape::Ref(_, _)) {
                    vec![ParamPtrPath {
                        param_idx,
                        fields: Vec::new(),
                    }]
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    fn init_entry_ref_aliases(
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        entry_params: &[(ValueId, Type)],
        arg_aliases: &[Vec<ParamPtrPath>],
    ) {
        for (param_idx, aliases) in arg_aliases.iter().enumerate() {
            if aliases.is_empty() {
                continue;
            }
            let value = entry_params[param_idx].0;
            let mapped = aliases
                .iter()
                .map(|alias| PtrPath {
                    root: entry_params[alias.param_idx].0,
                    fields: alias.fields.clone(),
                })
                .collect::<HashSet<_>>();
            ptr_aliases.insert(value, mapped);
        }
    }

    fn call_arg_aliases(
        args: &[ValueId],
        ptr_aliases: &HashMap<ValueId, HashSet<PtrPath>>,
    ) -> Vec<Vec<ParamPtrPath>> {
        let arg_aliases = args
            .iter()
            .map(|arg| ptr_aliases.get(arg).cloned().unwrap_or_default())
            .collect::<Vec<_>>();

        arg_aliases
            .iter()
            .map(|aliases| {
                let mut formal_aliases = HashSet::new();
                for alias in aliases {
                    for (param_idx, param_aliases) in arg_aliases.iter().enumerate() {
                        for param_alias in param_aliases {
                            if alias.root == param_alias.root
                                && alias.fields.starts_with(&param_alias.fields)
                            {
                                formal_aliases.insert(ParamPtrPath {
                                    param_idx,
                                    fields: alias.fields[param_alias.fields.len()..].to_vec(),
                                });
                            }
                        }
                    }
                }
                let mut formal_aliases = formal_aliases.into_iter().collect::<Vec<_>>();
                formal_aliases.sort();
                formal_aliases
            })
            .collect()
    }

    fn init_ref_location(
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        value: ValueId,
        wt: &WitnessShape,
    ) {
        let WitnessShape::Ref(_, inner) = wt else {
            return;
        };
        ptr_aliases
            .entry(value)
            .or_insert_with(|| HashSet::from([PtrPath::root(value)]));
        memory_wt
            .entry(value)
            .and_modify(|existing| *existing = existing.join(inner))
            .or_insert_with(|| *inner.clone());
    }

    fn ensure_ref_aliases(
        value_wt: &HashMap<ValueId, WitnessShape>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        value: ValueId,
    ) -> HashSet<PtrPath> {
        let wt = value_wt
            .get(&value)
            .unwrap_or_else(|| panic!("Missing witness type for ref value {:?}", value));
        Self::init_ref_location(ptr_aliases, memory_wt, value, wt);
        ptr_aliases
            .get(&value)
            .cloned()
            .unwrap_or_else(|| panic!("Missing pointer aliases for ref value {:?}", value))
    }

    fn join_ref_aliases(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        target: ValueId,
        source: ValueId,
    ) {
        let source_aliases = Self::ensure_ref_aliases(value_wt, ptr_aliases, memory_wt, source);
        Self::ensure_ref_aliases(value_wt, ptr_aliases, memory_wt, target);
        ptr_aliases
            .entry(target)
            .or_default()
            .extend(source_aliases);
        Self::sync_ref_value(value_wt, ptr_aliases, memory_wt, target);
    }

    fn join_ref_inner(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr: ValueId,
        new_inner: &WitnessShape,
    ) {
        let aliases = Self::ensure_ref_aliases(value_wt, ptr_aliases, memory_wt, ptr);
        for alias in aliases {
            Self::join_path(memory_wt, &alias, new_inner);
        }
        Self::sync_all_ref_values(value_wt, ptr_aliases, memory_wt);
    }

    fn read_ref_inner(
        value_wt: &HashMap<ValueId, WitnessShape>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr: ValueId,
    ) -> WitnessShape {
        let aliases = Self::ensure_ref_aliases(value_wt, ptr_aliases, memory_wt, ptr);
        let ptr_info = match value_wt.get(&ptr).unwrap() {
            WitnessShape::Ref(ptr_info, _) => *ptr_info,
            _ => panic!("Load from non-ref type"),
        };
        let inner = Self::read_aliases(memory_wt, &aliases);
        inner.with_toplevel_info(inner.toplevel_info().join(ptr_info))
    }

    fn propagate_ref_tuple_splice(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        tuple_ref: ValueId,
        result: ValueId,
        field_idx: usize,
    ) {
        let tuple_aliases = Self::ensure_ref_aliases(value_wt, ptr_aliases, memory_wt, tuple_ref);
        let field_aliases = tuple_aliases
            .iter()
            .map(|alias| alias.field(field_idx))
            .collect::<HashSet<_>>();
        let ptr_info = match value_wt.get(&tuple_ref).unwrap() {
            WitnessShape::Ref(ptr_info, _) => *ptr_info,
            _ => panic!("RefTupleSplice from non-ref witness type"),
        };
        let child = Self::read_aliases(memory_wt, &field_aliases);
        let result_wt = value_wt
            .get(&result)
            .map(|existing| match existing {
                WitnessShape::Ref(existing_info, existing_child) => WitnessShape::Ref(
                    existing_info.join(ptr_info),
                    Box::new(existing_child.join(&child)),
                ),
                other => panic!(
                    "RefTupleSplice result has non-ref witness type: {:?}",
                    other
                ),
            })
            .unwrap_or_else(|| WitnessShape::Ref(ptr_info, Box::new(child)));
        value_wt.insert(result, result_wt);

        ptr_aliases.entry(result).or_default().extend(field_aliases);
        Self::sync_ref_value(value_wt, ptr_aliases, memory_wt, result);
    }

    fn return_param_aliases(
        value: ValueId,
        ptr_aliases: &HashMap<ValueId, HashSet<PtrPath>>,
        entry_param_indices: &HashMap<ValueId, usize>,
    ) -> HashSet<ParamPtrPath> {
        ptr_aliases
            .get(&value)
            .into_iter()
            .flat_map(|aliases| aliases.iter())
            .filter_map(|alias| {
                entry_param_indices
                    .get(&alias.root)
                    .map(|param_idx| ParamPtrPath {
                        param_idx: *param_idx,
                        fields: alias.fields.clone(),
                    })
            })
            .collect()
    }

    fn apply_return_aliases(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr_aliases: &mut HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        result: ValueId,
        return_aliases: &HashSet<ParamPtrPath>,
        args: &[ValueId],
    ) {
        if return_aliases.is_empty() {
            return;
        }

        let mut mapped_aliases = HashSet::new();
        for returned_alias in return_aliases {
            let arg = args[returned_alias.param_idx];
            let arg_aliases = Self::ensure_ref_aliases(value_wt, ptr_aliases, memory_wt, arg);
            mapped_aliases.extend(
                arg_aliases
                    .iter()
                    .map(|alias| alias.extend(&returned_alias.fields)),
            );
        }

        ptr_aliases
            .entry(result)
            .or_default()
            .extend(mapped_aliases);
        Self::sync_ref_value(value_wt, ptr_aliases, memory_wt, result);
    }

    fn read_aliases(
        memory_wt: &HashMap<ValueId, WitnessShape>,
        aliases: &HashSet<PtrPath>,
    ) -> WitnessShape {
        let mut reads = aliases
            .iter()
            .map(|alias| Self::read_path(memory_wt, alias));
        let first = reads
            .next()
            .unwrap_or_else(|| panic!("Cannot read through empty pointer alias set"));
        reads.fold(first, |acc, next| acc.join(&next))
    }

    fn read_path(memory_wt: &HashMap<ValueId, WitnessShape>, path: &PtrPath) -> WitnessShape {
        let root = memory_wt
            .get(&path.root)
            .unwrap_or_else(|| panic!("Missing memory witness shape for root {:?}", path.root));
        Self::read_shape_path(root, &path.fields)
    }

    fn read_shape_path(shape: &WitnessShape, fields: &[usize]) -> WitnessShape {
        let Some((&field_idx, rest)) = fields.split_first() else {
            return shape.clone();
        };
        match shape {
            WitnessShape::Tuple(top, children) => {
                let child = Self::read_shape_path(&children[field_idx], rest);
                child.with_toplevel_info(child.toplevel_info().join(*top))
            }
            other => panic!(
                "Cannot read tuple field path {:?} through non-tuple witness type {:?}",
                fields, other
            ),
        }
    }

    fn join_path(
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        path: &PtrPath,
        new_wt: &WitnessShape,
    ) {
        let root = memory_wt
            .get_mut(&path.root)
            .unwrap_or_else(|| panic!("Missing memory witness shape for root {:?}", path.root));
        Self::join_shape_path(root, &path.fields, new_wt);
    }

    fn join_shape_path(shape: &mut WitnessShape, fields: &[usize], new_wt: &WitnessShape) {
        let Some((&field_idx, rest)) = fields.split_first() else {
            *shape = shape.join(new_wt);
            return;
        };
        match shape {
            WitnessShape::Tuple(_, children) => {
                Self::join_shape_path(&mut children[field_idx], rest, new_wt);
            }
            other => panic!(
                "Cannot write tuple field path {:?} through non-tuple witness type {:?}",
                fields, other
            ),
        }
    }

    fn sync_all_ref_values(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr_aliases: &HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
    ) {
        let values = ptr_aliases.keys().cloned().collect::<Vec<_>>();
        for value in values {
            Self::sync_ref_value(value_wt, ptr_aliases, memory_wt, value);
        }
    }

    fn sync_ref_value(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr_aliases: &HashMap<ValueId, HashSet<PtrPath>>,
        memory_wt: &mut HashMap<ValueId, WitnessShape>,
        value: ValueId,
    ) {
        let Some(aliases) = ptr_aliases.get(&value) else {
            return;
        };
        let Some(WitnessShape::Ref(ptr_info, inner)) = value_wt.get(&value).cloned() else {
            return;
        };
        for alias in aliases {
            Self::join_path(memory_wt, alias, &inner);
        }
        let alias_inner = Self::read_aliases(memory_wt, aliases);
        value_wt.insert(
            value,
            WitnessShape::Ref(ptr_info, Box::new(inner.join(&alias_inner))),
        );
    }

    fn build_function_witness_type(
        spec_key: &SpecKey,
        spec_value: &SpecValue,
        value_wt: &HashMap<ValueId, WitnessShape>,
        block_cfg: &HashMap<BlockId, WitnessType>,
    ) -> FunctionWitnessType {
        FunctionWitnessType {
            returns_witness: spec_value.return_types.clone(),
            cfg_witness: spec_key.cfg_witness,
            parameters: spec_key.arg_types.clone(),
            block_cfg_witness: block_cfg.clone(),
            value_witness_types: value_wt.clone(),
        }
    }

    fn infer_binop_cmp_result(lhs_wt: &WitnessShape, rhs_wt: &WitnessShape) -> WitnessShape {
        WitnessShape::Scalar(lhs_wt.toplevel_info().join(rhs_wt.toplevel_info()))
    }

    fn infer_select_result(
        cond_wt: &WitnessShape,
        then_wt: &WitnessShape,
        otherwise_wt: &WitnessShape,
    ) -> WitnessShape {
        then_wt.join(otherwise_wt).with_toplevel_info(
            cond_wt
                .toplevel_info()
                .join(then_wt.toplevel_info())
                .join(otherwise_wt.toplevel_info()),
        )
    }

    fn construct_pure_witness_for_type(typ: &Type) -> WitnessShape {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field => {
                WitnessShape::Scalar(WitnessType::Pure)
            }
            TypeExpr::Array(i, _) => WitnessShape::Array(
                WitnessType::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Slice(i) => WitnessShape::Array(
                WitnessType::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Ref(i) => WitnessShape::Ref(
                WitnessType::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::WitnessOf(_) => {
                panic!("ICE: WitnessOf should not be present at this stage");
            }
            TypeExpr::Tuple(elements) => WitnessShape::Tuple(
                WitnessType::Pure,
                elements
                    .iter()
                    .map(Self::construct_pure_witness_for_type)
                    .collect(),
            ),
            TypeExpr::Function => WitnessShape::Scalar(WitnessType::Pure),
        }
    }
}

impl SSAAnotator for WitnessTypeInference {
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
