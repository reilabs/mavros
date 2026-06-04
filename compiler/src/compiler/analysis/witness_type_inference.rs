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
    cfg_witness: WitnessType,
}

#[derive(Clone, Debug)]
struct SpecValue {
    specialized_func_id: FunctionId,
    return_types: Vec<WitnessShape>,
    return_constraints: Vec<WitnessShape>,
}

// ---------------------------------------------------------------------------
// Per-function propagation result
// ---------------------------------------------------------------------------

struct PropagationResult {
    return_types: Vec<WitnessShape>,
    arg_types: Vec<WitnessShape>,
    value_wt: HashMap<ValueId, WitnessShape>,
    block_cfg: HashMap<BlockId, WitnessType>,
    /// Call sites discovered after local propagation has reached a fixed point.
    call_sites: Vec<CallSiteInfo>,
}

#[derive(Clone)]
struct CallSiteInfo {
    callee_func_id: FunctionId,
    arg_types: Vec<WitnessShape>,
    result_types: Vec<WitnessShape>,
    cfg_witness: WitnessType,
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

        let main_return_types: Vec<WitnessShape> = main_func
            .get_returns()
            .iter()
            .map(Self::construct_pure_witness_for_type)
            .collect();

        let main_cfg_witness = WitnessType::Pure;

        // 2. Register main specialization
        let main_key = SpecKey {
            original_func_id: main_id,
            arg_types: main_arg_types.clone(),
            cfg_witness: main_cfg_witness,
        };

        // Clone main function as specialized version, set as entry point
        let main_specialized_id = ssa.duplicate_function(main_id);
        ssa.set_entry_point(main_specialized_id);

        let mut specializations: HashMap<SpecKey, SpecValue> = HashMap::new();
        let mut worklist: VecDeque<SpecKey> = VecDeque::new();
        let mut queued: HashSet<SpecKey> = HashSet::new();
        let mut redirects: HashMap<SpecKey, SpecKey> = HashMap::new();
        // Track which specializations call which (for re-queuing callers)
        let mut callers: HashMap<SpecKey, HashSet<SpecKey>> = HashMap::new();
        let join_shapes = |lhs: &[WitnessShape], rhs: &[WitnessShape]| -> Vec<WitnessShape> {
            assert_eq!(
                lhs.len(),
                rhs.len(),
                "Cannot join witness shape vectors of different lengths"
            );
            lhs.iter().zip(rhs.iter()).map(|(a, b)| a.join(b)).collect()
        };

        specializations.insert(
            main_key.clone(),
            SpecValue {
                specialized_func_id: main_specialized_id,
                return_types: main_return_types.clone(),
                return_constraints: main_return_types,
            },
        );
        worklist.push_back(main_key.clone());
        queued.insert(main_key.clone());

        // 3. Global worklist loop
        while let Some(queued_key) = worklist.pop_front() {
            queued.remove(&queued_key);
            let spec_key = Self::resolve_key(&queued_key, &redirects);
            queued.remove(&spec_key);
            let Some(spec_value_snapshot) = specializations.get(&spec_key).cloned() else {
                continue;
            };

            let specialized_func_id = spec_value_snapshot.specialized_func_id;
            let return_constraints = spec_value_snapshot.return_constraints;
            let func = ssa.get_function(specialized_func_id);
            let cfg = flow_analysis.get_function_cfg(spec_key.original_func_id);

            let result = Self::propagate_function(
                func,
                cfg,
                &spec_key.arg_types,
                &return_constraints,
                spec_key.cfg_witness,
                &specializations,
                &redirects,
                ssa,
            );

            let closed_arg_types = join_shapes(&spec_key.arg_types, &result.arg_types);
            if closed_arg_types != spec_key.arg_types {
                let closed_key = SpecKey {
                    original_func_id: spec_key.original_func_id,
                    arg_types: closed_arg_types,
                    cfg_witness: spec_key.cfg_witness,
                };

                let mut old_value = specializations.remove(&spec_key).unwrap();
                old_value.return_types = join_shapes(&old_value.return_types, &result.return_types);
                redirects.insert(spec_key.clone(), closed_key.clone());
                self.functions.remove(&old_value.specialized_func_id);

                if let Some(old_callers) = callers.remove(&spec_key) {
                    callers
                        .entry(closed_key.clone())
                        .or_default()
                        .extend(old_callers);
                }

                if let Some(existing) = specializations.get_mut(&closed_key) {
                    existing.return_types =
                        join_shapes(&existing.return_types, &old_value.return_types);
                    existing.return_constraints =
                        join_shapes(&existing.return_constraints, &old_value.return_constraints);
                } else {
                    specializations.insert(closed_key.clone(), old_value);
                }

                Self::enqueue_key(&mut worklist, &mut queued, closed_key.clone(), &redirects);
                Self::enqueue_callers(
                    &mut worklist,
                    &mut queued,
                    &callers,
                    &closed_key,
                    &redirects,
                );
                continue;
            }

            let spec_value = specializations.get_mut(&spec_key).unwrap();
            let changed = result.return_types != spec_value.return_types;
            spec_value.return_types = result.return_types;

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
                let callee_key = Self::resolve_key(
                    &SpecKey {
                        original_func_id: call_site.callee_func_id,
                        arg_types: call_site.arg_types.clone(),
                        cfg_witness: call_site.cfg_witness,
                    },
                    &redirects,
                );

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

                    let specialized_id = ssa.duplicate_function(callee_key.original_func_id);

                    specializations.insert(
                        callee_key.clone(),
                        SpecValue {
                            specialized_func_id: specialized_id,
                            return_types: callee_return_types.clone(),
                            return_constraints: callee_return_types,
                        },
                    );
                    Self::enqueue_key(&mut worklist, &mut queued, callee_key.clone(), &redirects);
                }

                let callee_spec = specializations.get_mut(&callee_key).unwrap();
                assert_eq!(
                    callee_spec.return_constraints.len(),
                    call_site.result_types.len(),
                    "Cannot join return constraints of different lengths"
                );
                let new_constraints: Vec<WitnessShape> = callee_spec
                    .return_constraints
                    .iter()
                    .zip(call_site.result_types.iter())
                    .map(|(current, seen)| {
                        if Self::contains_ref(current) || Self::contains_ref(seen) {
                            current.join(seen)
                        } else {
                            current.clone()
                        }
                    })
                    .collect();
                if new_constraints != callee_spec.return_constraints {
                    callee_spec.return_constraints = new_constraints;
                    Self::enqueue_key(&mut worklist, &mut queued, callee_key, &redirects);
                }
            }

            // If output changed, re-queue callers
            if changed {
                Self::enqueue_callers(&mut worklist, &mut queued, &callers, &spec_key, &redirects);
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
                &spec_value.return_constraints,
                spec_key.cfg_witness,
                &specializations,
                &redirects,
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
                        let callee_key = Self::resolve_key(
                            &SpecKey {
                                original_func_id: call_site.callee_func_id,
                                arg_types: call_site.arg_types.clone(),
                                cfg_witness: call_site.cfg_witness,
                            },
                            &redirects,
                        );
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
        return_constraints: &[WitnessShape],
        cfg_witness: WitnessType,
        specializations: &HashMap<SpecKey, SpecValue>,
        redirects: &HashMap<SpecKey, SpecKey>,
        ssa: &HLSSA,
    ) -> PropagationResult {
        let entry_id = func.get_entry_id();
        let block_queue: Vec<BlockId> = cfg.get_blocks_bfs().collect();

        // Inner state
        let mut value_wt: HashMap<ValueId, WitnessShape> = HashMap::new();
        let mut block_cfg: HashMap<BlockId, WitnessType> = HashMap::new();

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
                    let wt = Self::construct_pure_witness_for_type(tp);
                    value_wt.insert(*value_id, wt);
                }
            }
        }

        // Inner iteration until stable
        let max_iterations = 100;
        for _iteration in 0..max_iterations {
            let old_value_wt = value_wt.clone();
            let old_block_cfg = block_cfg.clone();

            Self::propagate_once(
                func,
                cfg,
                &block_queue,
                entry_id,
                return_constraints,
                &mut value_wt,
                &mut block_cfg,
                specializations,
                redirects,
                ssa,
            );

            if value_wt == old_value_wt && block_cfg == old_block_cfg {
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
                    results,
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
                    let callee_key = Self::resolve_key(
                        &SpecKey {
                            original_func_id: *callee_id,
                            arg_types: callee_arg_types,
                            cfg_witness: block_cw,
                        },
                        redirects,
                    );
                    let result_types: Vec<WitnessShape> = results
                        .iter()
                        .map(|v| value_wt.get(v).unwrap().clone())
                        .collect();
                    call_sites.push(CallSiteInfo {
                        callee_func_id: callee_key.original_func_id,
                        arg_types: callee_key.arg_types,
                        result_types,
                        cfg_witness: callee_key.cfg_witness,
                    });
                }
            }

            if let Some(Terminator::Return(values)) = block.get_terminator() {
                let ret_wts: Vec<WitnessShape> = values
                    .iter()
                    .zip(return_constraints.iter())
                    .map(|(v, constraint)| value_wt.get(v).unwrap().join(constraint))
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
                .map(Self::construct_pure_witness_for_type)
                .collect();
        }

        // Compute closed argument types: for values that may contain references,
        // reflect any bidirectional alias updates discovered inside the function.
        let arg_types: Vec<WitnessShape> = entry_params
            .iter()
            .zip(arg_types.iter())
            .map(|((value_id, _), original_arg)| match original_arg {
                _ if Self::contains_ref(original_arg) => value_wt
                    .get(value_id)
                    .cloned()
                    .unwrap_or_else(|| original_arg.clone()),
                _ => original_arg.clone(),
            })
            .collect();

        PropagationResult {
            return_types,
            arg_types,
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
        return_constraints: &[WitnessShape],
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        block_cfg: &mut HashMap<BlockId, WitnessType>,
        specializations: &HashMap<SpecKey, SpecValue>,
        redirects: &HashMap<SpecKey, SpecKey>,
        ssa: &HLSSA,
    ) {
        for block_id in block_queue {
            let block = func.get_block(*block_id);
            let block_cw = *block_cfg.get(block_id).unwrap();

            Self::propagate_block_instructions(
                block,
                block_cw,
                value_wt,
                specializations,
                redirects,
                ssa,
            );

            // Handle terminator
            if let Some(terminator) = block.get_terminator() {
                match terminator {
                    Terminator::Return(values) => {
                        for (value, constraint) in values.iter().zip(return_constraints.iter()) {
                            if Self::contains_ref(constraint) {
                                Self::merge_value_wt(value_wt, *value, constraint.clone());
                            }
                        }
                    }
                    Terminator::Jmp(target, params) => {
                        let target_params: Vec<(ValueId, Type)> =
                            func.get_block(*target).get_parameters().cloned().collect();
                        for ((target_value, _), param) in target_params.iter().zip(params.iter()) {
                            let param_wt = value_wt.get(param).unwrap().clone();
                            let existing = value_wt.get(target_value).unwrap().clone();
                            let joined = existing.join(&param_wt);
                            value_wt.insert(*target_value, joined.clone());
                            if Self::contains_ref(&joined) {
                                Self::merge_value_wt(value_wt, *param, joined);
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
    }

    /// Process all instructions in a single block, updating `value_wt` according
    /// to each opcode's witness-type rules.
    fn propagate_block_instructions(
        block: &HLBlock,
        block_cw: WitnessType,
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        specializations: &HashMap<SpecKey, SpecValue>,
        redirects: &HashMap<SpecKey, SpecKey>,
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
                    let result_wt = value_wt.get(r).unwrap().clone();
                    if Self::contains_ref(&result_wt) {
                        Self::merge_value_wt(value_wt, *if_t, result_wt.clone());
                        Self::merge_value_wt(value_wt, *if_f, result_wt);
                    }
                }
                OpCode::Alloc {
                    result: r,
                    elem_type: tp,
                } => {
                    let inner = Self::construct_pure_witness_for_type(tp);
                    let wt = WitnessShape::Ref(WitnessType::Pure, Box::new(inner));
                    value_wt.insert(*r, wt.clone());
                }
                OpCode::Store { ptr, value: v } => {
                    let val_wt = value_wt.get(v).unwrap().clone();
                    // cfg_witness contributes to the toplevel of what's stored.
                    let store_wt = val_wt.with_toplevel_info(val_wt.toplevel_info().join(block_cw));
                    Self::merge_ref_inner(value_wt, *ptr, store_wt.clone());
                    if Self::contains_ref(&store_wt) {
                        let stored_wt = Self::read_ref_inner(value_wt, *ptr);
                        Self::merge_value_wt(value_wt, *v, stored_wt);
                    }
                }
                OpCode::Load { result: r, ptr } => {
                    let result_wt = Self::read_ref_inner(value_wt, *ptr);
                    Self::merge_value_wt(value_wt, *r, result_wt);
                    let result_wt = value_wt.get(r).unwrap().clone();
                    if Self::contains_ref(&result_wt) {
                        Self::merge_ref_inner(value_wt, *ptr, result_wt);
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
                    Self::merge_value_wt(value_wt, *r, result_wt);
                    let result_wt = value_wt.get(r).unwrap().clone();
                    if Self::contains_ref(&result_wt) {
                        let arr_wt = value_wt.get(arr).unwrap().clone();
                        let updated_arr_wt = match arr_wt {
                            WitnessShape::Array(array_info, elem) => {
                                WitnessShape::Array(array_info, Box::new(elem.join(&result_wt)))
                            }
                            other => {
                                panic!("ArrayGet on non-array witness type: {:?}", other);
                            }
                        };
                        Self::merge_value_wt(value_wt, *arr, updated_arr_wt);
                    }
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
                    Self::merge_value_wt(value_wt, *r, result_wt);
                    let result_elem_wt = value_wt.get(r).unwrap().child_witness_type().unwrap();
                    if Self::contains_ref(&result_elem_wt) {
                        let arr_wt = value_wt.get(arr).unwrap().clone();
                        let updated_arr_wt = match arr_wt {
                            WitnessShape::Array(array_info, elem) => WitnessShape::Array(
                                array_info,
                                Box::new(elem.join(&result_elem_wt)),
                            ),
                            other => {
                                panic!("ArraySet on non-array witness type: {:?}", other);
                            }
                        };
                        Self::merge_value_wt(value_wt, *arr, updated_arr_wt);
                        Self::merge_value_wt(value_wt, *value, result_elem_wt);
                    }
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
                    Self::merge_value_wt(value_wt, *r, result_wt);
                    let result_elem_wt = value_wt.get(r).unwrap().child_witness_type().unwrap();
                    if Self::contains_ref(&result_elem_wt) {
                        let slice_wt = value_wt.get(sl).unwrap().clone();
                        let updated_slice_wt = match slice_wt {
                            WitnessShape::Array(slice_info, elem) => WitnessShape::Array(
                                slice_info,
                                Box::new(elem.join(&result_elem_wt)),
                            ),
                            other => {
                                panic!("SlicePush on non-array witness type: {:?}", other);
                            }
                        };
                        Self::merge_value_wt(value_wt, *sl, updated_slice_wt);
                        for value in values {
                            Self::merge_value_wt(value_wt, *value, result_elem_wt.clone());
                        }
                    }
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
                        }
                    } else {
                        let callee_arg_types: Vec<WitnessShape> = args
                            .iter()
                            .map(|v| value_wt.get(v).unwrap().clone())
                            .collect();
                        let callee_key = Self::resolve_key(
                            &SpecKey {
                                original_func_id: *callee_id,
                                arg_types: callee_arg_types,
                                cfg_witness: block_cw,
                            },
                            redirects,
                        );

                        if let Some(callee_spec) = specializations.get(&callee_key) {
                            // Use callee's return types
                            for (result, ret_wt) in
                                results.iter().zip(callee_spec.return_types.iter())
                            {
                                Self::merge_value_wt(value_wt, *result, ret_wt.clone());
                            }
                            // A resolved key is closed over by-reference argument effects.
                            for (arg, arg_wt) in args.iter().zip(callee_key.arg_types.iter()) {
                                Self::merge_value_wt(value_wt, *arg, arg_wt.clone());
                            }
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
                    Self::merge_value_wt(
                        value_wt,
                        *result,
                        WitnessShape::Array(WitnessType::Pure, Box::new(result_wt)),
                    );
                    let result_elem_wt =
                        value_wt.get(result).unwrap().child_witness_type().unwrap();
                    if Self::contains_ref(&result_elem_wt) {
                        for elem in elems {
                            Self::merge_value_wt(value_wt, *elem, result_elem_wt.clone());
                        }
                    }
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
                    Self::merge_value_wt(
                        value_wt,
                        *result,
                        WitnessShape::Array(WitnessType::Pure, Box::new(result_wt)),
                    );
                    let result_elem_wt =
                        value_wt.get(result).unwrap().child_witness_type().unwrap();
                    if Self::contains_ref(&result_elem_wt) {
                        Self::merge_value_wt(value_wt, *element, result_elem_wt);
                    }
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
                            let result_wt = value_wt.get(result).unwrap().clone();
                            if Self::contains_ref(&result_wt) {
                                let tuple_wt = value_wt.get(tuple).unwrap().clone();
                                let updated_tuple_wt = match tuple_wt {
                                    WitnessShape::Tuple(tuple_info, mut children) => {
                                        children[*idx] = children[*idx].join(&result_wt);
                                        WitnessShape::Tuple(tuple_info, children)
                                    }
                                    other => {
                                        panic!("TupleProj on non-tuple witness type: {:?}", other);
                                    }
                                };
                                Self::merge_value_wt(value_wt, *tuple, updated_tuple_wt);
                            }
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
                    let result_wt = value_wt.get(result).unwrap().clone();
                    if let WitnessShape::Tuple(_, children) = result_wt {
                        for (elem, child_wt) in elems.iter().zip(children.iter()) {
                            if Self::contains_ref(child_wt) {
                                Self::merge_value_wt(value_wt, *elem, child_wt.clone());
                            }
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
                _ => {
                    panic!(
                        "Unsupported opcode during witness type inference: {:?}",
                        instruction
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn resolve_key(key: &SpecKey, redirects: &HashMap<SpecKey, SpecKey>) -> SpecKey {
        let mut current = key.clone();
        let mut seen = HashSet::new();
        while let Some(next) = redirects.get(&current) {
            assert!(
                seen.insert(current.clone()),
                "Cycle in witness specialization redirects at {:?}",
                current
            );
            current = next.clone();
        }
        current
    }

    fn enqueue_key(
        worklist: &mut VecDeque<SpecKey>,
        queued: &mut HashSet<SpecKey>,
        key: SpecKey,
        redirects: &HashMap<SpecKey, SpecKey>,
    ) {
        let key = Self::resolve_key(&key, redirects);
        if queued.insert(key.clone()) {
            worklist.push_back(key);
        }
    }

    fn enqueue_callers(
        worklist: &mut VecDeque<SpecKey>,
        queued: &mut HashSet<SpecKey>,
        callers: &HashMap<SpecKey, HashSet<SpecKey>>,
        key: &SpecKey,
        redirects: &HashMap<SpecKey, SpecKey>,
    ) {
        if let Some(caller_set) = callers.get(key) {
            for caller_key in caller_set {
                Self::enqueue_key(worklist, queued, caller_key.clone(), redirects);
            }
        }
    }

    fn contains_ref(wt: &WitnessShape) -> bool {
        match wt {
            WitnessShape::Ref(_, _) => true,
            WitnessShape::Array(_, inner) => Self::contains_ref(inner),
            WitnessShape::Tuple(_, children) => children.iter().any(Self::contains_ref),
            WitnessShape::Scalar(_) => false,
        }
    }

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

    fn read_ref_inner(value_wt: &HashMap<ValueId, WitnessShape>, ptr: ValueId) -> WitnessShape {
        match value_wt.get(&ptr).unwrap() {
            WitnessShape::Ref(ptr_info, inner) => {
                inner.with_toplevel_info(inner.toplevel_info().join(*ptr_info))
            }
            other => panic!("Load from non-ref witness type: {:?}", other),
        }
    }

    fn merge_ref_inner(
        value_wt: &mut HashMap<ValueId, WitnessShape>,
        ptr: ValueId,
        new_inner: WitnessShape,
    ) {
        let ptr_wt = value_wt.get(&ptr).unwrap().clone();
        let updated = match ptr_wt {
            WitnessShape::Ref(ptr_info, inner) => {
                WitnessShape::Ref(ptr_info, Box::new(inner.join(&new_inner)))
            }
            other => panic!("Store to non-ref witness type: {:?}", other),
        };
        Self::merge_value_wt(value_wt, ptr, updated);
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
