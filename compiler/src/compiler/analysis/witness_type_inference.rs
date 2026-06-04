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

struct PropagationState {
    value_wt: HashMap<ValueId, WitnessShape>,
    block_cfg: HashMap<BlockId, WitnessType>,
}

impl PropagationState {
    fn witness(&self, value: ValueId) -> &WitnessShape {
        self.value_wt.get(&value).unwrap()
    }

    fn join_value(&mut self, value: ValueId, wt: WitnessShape) -> WitnessShape {
        let joined = self
            .value_wt
            .get(&value)
            .map(|existing| existing.join(&wt))
            .unwrap_or(wt);
        self.value_wt.insert(value, joined.clone());
        joined
    }

    fn ref_inner(&self, ptr: ValueId) -> WitnessShape {
        match self.witness(ptr) {
            WitnessShape::Ref(ptr_info, inner) => {
                inner.with_toplevel_info(inner.toplevel_info().join(*ptr_info))
            }
            other => panic!("Load from non-ref witness type: {:?}", other),
        }
    }

    fn join_ref_inner(&mut self, ptr: ValueId, inner_wt: WitnessShape) {
        let ptr_wt = self.witness(ptr).clone();
        let updated = match ptr_wt {
            WitnessShape::Ref(ptr_info, inner) => {
                WitnessShape::Ref(ptr_info, Box::new(inner.join(&inner_wt)))
            }
            other => panic!("Store to non-ref witness type: {:?}", other),
        };
        self.join_value(ptr, updated);
    }

    fn back_value(&mut self, value: ValueId, wt: &WitnessShape) {
        if wt.contains_ref() {
            self.join_value(value, wt.clone());
        }
    }

    fn back_values(&mut self, values: &[ValueId], wt: &WitnessShape) {
        if wt.contains_ref() {
            for value in values {
                self.join_value(*value, wt.clone());
            }
        }
    }

    fn back_ref_inner(&mut self, ptr: ValueId, wt: &WitnessShape) {
        if wt.contains_ref() {
            self.join_ref_inner(ptr, wt.clone());
        }
    }

    fn array_element(&self, array: ValueId) -> WitnessShape {
        match self.witness(array) {
            WitnessShape::Array(_, elem) => *elem.clone(),
            other => panic!(
                "Array element access on non-array witness type: {:?}",
                other
            ),
        }
    }

    fn back_array_element(&mut self, array: ValueId, wt: &WitnessShape) {
        if wt.contains_ref() {
            let updated = match self.witness(array).clone() {
                WitnessShape::Array(array_info, elem) => {
                    WitnessShape::Array(array_info, Box::new(elem.join(wt)))
                }
                other => panic!("Array element merge on non-array witness type: {:?}", other),
            };
            self.join_value(array, updated);
        }
    }

    fn tuple_element(&self, tuple: ValueId, idx: usize) -> (WitnessType, WitnessShape) {
        match self.witness(tuple) {
            WitnessShape::Tuple(tuple_info, children) => (*tuple_info, children[idx].clone()),
            other => panic!(
                "Tuple element access on non-tuple witness type: {:?}",
                other
            ),
        }
    }

    fn back_tuple_element(&mut self, tuple: ValueId, idx: usize, wt: &WitnessShape) {
        if wt.contains_ref() {
            let updated = match self.witness(tuple).clone() {
                WitnessShape::Tuple(tuple_info, mut children) => {
                    children[idx] = children[idx].join(wt);
                    WitnessShape::Tuple(tuple_info, children)
                }
                other => panic!("Tuple element merge on non-tuple witness type: {:?}", other),
            };
            self.join_value(tuple, updated);
        }
    }
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
                        if current.contains_ref() || seen.contains_ref() {
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

        let mut state = PropagationState {
            value_wt: HashMap::new(),
            block_cfg: HashMap::new(),
        };

        ssa.for_each_const(|vid, val| {
            let shape = match val.as_ref() {
                Constant::U(_, _) | Constant::I(_, _) | Constant::Field(_) | Constant::FnPtr(_) => {
                    WitnessShape::Scalar(WitnessType::Pure)
                }
            };
            state.value_wt.insert(*vid, shape);
        });

        // Initialize entry block params from arg_types
        let entry_params: Vec<(ValueId, Type)> =
            func.get_entry().get_parameters().cloned().collect();
        for ((value_id, _), wt) in entry_params.iter().zip(arg_types.iter()) {
            state.value_wt.insert(*value_id, wt.clone());
        }

        // Initialize other block params as Pure (optimistic)
        for (block_id, block) in func.get_blocks() {
            if *block_id == entry_id {
                state.block_cfg.insert(*block_id, cfg_witness);
                continue;
            }
            state.block_cfg.insert(*block_id, cfg_witness); // function minimum
            for (value_id, tp) in block.get_parameters() {
                if !state.value_wt.contains_key(value_id) {
                    let wt = Self::construct_pure_witness_for_type(tp);
                    state.value_wt.insert(*value_id, wt);
                }
            }
        }

        // Inner iteration until stable
        let max_iterations = 100;
        for _iteration in 0..max_iterations {
            let old_value_wt = state.value_wt.clone();
            let old_block_cfg = state.block_cfg.clone();

            Self::propagate_once(
                func,
                cfg,
                &block_queue,
                entry_id,
                return_constraints,
                &mut state,
                specializations,
                redirects,
                ssa,
            );

            if state.value_wt == old_value_wt && state.block_cfg == old_block_cfg {
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
            let block_cw = *state.block_cfg.get(block_id).unwrap();

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
                    let callee_arg_types: Vec<WitnessShape> =
                        args.iter().map(|v| state.witness(*v).clone()).collect();
                    let callee_key = Self::resolve_key(
                        &SpecKey {
                            original_func_id: *callee_id,
                            arg_types: callee_arg_types,
                            cfg_witness: block_cw,
                        },
                        redirects,
                    );
                    let result_types: Vec<WitnessShape> =
                        results.iter().map(|v| state.witness(*v).clone()).collect();
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
                    .map(|(v, constraint)| state.witness(*v).join(constraint))
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
                _ if original_arg.contains_ref() => state
                    .value_wt
                    .get(value_id)
                    .cloned()
                    .unwrap_or_else(|| original_arg.clone()),
                _ => original_arg.clone(),
            })
            .collect();

        PropagationResult {
            return_types,
            arg_types,
            value_wt: state.value_wt,
            block_cfg: state.block_cfg,
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
        state: &mut PropagationState,
        specializations: &HashMap<SpecKey, SpecValue>,
        redirects: &HashMap<SpecKey, SpecKey>,
        ssa: &HLSSA,
    ) {
        for block_id in block_queue {
            let block = func.get_block(*block_id);
            let block_cw = *state.block_cfg.get(block_id).unwrap();

            Self::propagate_block_instructions(
                block,
                block_cw,
                state,
                specializations,
                redirects,
                ssa,
            );

            // Handle terminator
            if let Some(terminator) = block.get_terminator() {
                match terminator {
                    Terminator::Return(values) => {
                        for (value, constraint) in values.iter().zip(return_constraints.iter()) {
                            if constraint.contains_ref() {
                                state.join_value(*value, constraint.clone());
                            }
                        }
                    }
                    Terminator::Jmp(target, params) => {
                        let target_params: Vec<(ValueId, Type)> =
                            func.get_block(*target).get_parameters().cloned().collect();
                        for ((target_value, _), param) in target_params.iter().zip(params.iter()) {
                            let param_wt = state.witness(*param).clone();
                            let joined = state.join_value(*target_value, param_wt);
                            if joined.contains_ref() {
                                state.join_value(*param, joined);
                            }
                        }
                    }
                    Terminator::JmpIf(cond, _if_true, _if_false) => {
                        let cond_toplevel = state.witness(*cond).toplevel_info();
                        // Both loops and if-else: join cond into merge point params
                        // and body block cfg_witnesses
                        let merge = cfg.get_merge_point(*block_id);
                        let merge_params: Vec<ValueId> = func
                            .get_block(merge)
                            .get_parameters()
                            .map(|(v, _)| *v)
                            .collect();
                        for param_id in merge_params {
                            let existing = state.witness(param_id);
                            let joined = existing.with_witness_in_leaves(cond_toplevel);
                            state.value_wt.insert(param_id, joined);
                        }

                        let body_blocks = cfg.get_if_body(*block_id);
                        for body_block_id in body_blocks {
                            let existing_cfg = *state.block_cfg.get(&body_block_id).unwrap();
                            let new_cfg = existing_cfg.join(cond_toplevel).join(block_cw);
                            state.block_cfg.insert(body_block_id, new_cfg);
                        }
                    }
                }
            }
        }
    }

    /// Process all instructions in a single block.
    ///
    /// Each arm is a witness transfer rule:
    ///   - compute the value flowing into the result;
    ///   - join it into the monotone state;
    ///   - for containers/refs, add the reverse edge that makes aliases visible
    ///     at the specialization boundary.
    fn propagate_block_instructions(
        block: &HLBlock,
        block_cw: WitnessType,
        state: &mut PropagationState,
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
                    state.join_value(
                        *r,
                        Self::infer_binop_cmp_result(state.witness(*lhs), state.witness(*rhs)),
                    );
                }
                OpCode::Select {
                    result: r,
                    cond,
                    if_t,
                    if_f,
                } => {
                    let result_wt = Self::infer_select_result(
                        state.witness(*cond),
                        state.witness(*if_t),
                        state.witness(*if_f),
                    );
                    let result_wt = state.join_value(*r, result_wt);
                    state.back_values(&[*if_t, *if_f], &result_wt);
                }
                OpCode::Alloc {
                    result: r,
                    elem_type: tp,
                } => {
                    let inner = Self::construct_pure_witness_for_type(tp);
                    let wt = WitnessShape::Ref(WitnessType::Pure, Box::new(inner));
                    state.join_value(*r, wt);
                }
                OpCode::Store { ptr, value: v } => {
                    let val_wt = state.witness(*v).clone();
                    let store_wt = val_wt.with_toplevel_info(val_wt.toplevel_info().join(block_cw));
                    state.join_ref_inner(*ptr, store_wt.clone());
                    if store_wt.contains_ref() {
                        state.join_value(*v, state.ref_inner(*ptr));
                    }
                }
                OpCode::Load { result: r, ptr } => {
                    let result_wt = state.join_value(*r, state.ref_inner(*ptr));
                    state.back_ref_inner(*ptr, &result_wt);
                }
                OpCode::ReadGlobal {
                    result: r,
                    offset: _,
                    result_type: tp,
                } => {
                    state.join_value(*r, Self::construct_pure_witness_for_type(tp));
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
                    let array_top = state.witness(*arr).toplevel_info();
                    let index_top = state.witness(*idx).toplevel_info();
                    let result_wt = state
                        .array_element(*arr)
                        .with_witness_in_leaves(array_top.join(index_top));
                    let result_wt = state.join_value(*r, result_wt);
                    state.back_array_element(*arr, &result_wt);
                }
                OpCode::ArraySet {
                    result: r,
                    array: arr,
                    index: idx,
                    value,
                } => {
                    let arr_top = state.witness(*arr).toplevel_info();
                    let idx_top = state.witness(*idx).toplevel_info();
                    let arr_elem_wt = state.array_element(*arr);
                    let val_wt = state.witness(*value);
                    let result_elem_wt = arr_elem_wt.join(val_wt).with_toplevel_info(
                        arr_elem_wt
                            .toplevel_info()
                            .join(val_wt.toplevel_info())
                            .join(idx_top),
                    );
                    state.join_value(*r, WitnessShape::Array(arr_top, Box::new(result_elem_wt)));
                    let result_elem_wt = state.array_element(*r);
                    state.back_array_element(*arr, &result_elem_wt);
                    state.back_value(*value, &result_elem_wt);
                }
                OpCode::SlicePush {
                    dir: _,
                    result: r,
                    slice: sl,
                    values,
                } => {
                    let slice_top = state.witness(*sl).toplevel_info();
                    let mut result_elem_wt = state.array_element(*sl);
                    for value in values {
                        result_elem_wt = result_elem_wt.join(state.witness(*value));
                    }
                    state.join_value(*r, WitnessShape::Array(slice_top, Box::new(result_elem_wt)));
                    let result_elem_wt = state.array_element(*r);
                    state.back_array_element(*sl, &result_elem_wt);
                    state.back_values(values, &result_elem_wt);
                }
                OpCode::SliceLen {
                    result: r,
                    slice: _,
                } => {
                    state.join_value(*r, WitnessShape::Scalar(WitnessType::Pure));
                }
                OpCode::Call {
                    results,
                    function: CallTarget::Static(callee_id),
                    args,
                    unconstrained,
                } => {
                    let callee_key = if *unconstrained {
                        None
                    } else {
                        let callee_arg_types: Vec<WitnessShape> =
                            args.iter().map(|v| state.witness(*v).clone()).collect();
                        Some(Self::resolve_key(
                            &SpecKey {
                                original_func_id: *callee_id,
                                arg_types: callee_arg_types,
                                cfg_witness: block_cw,
                            },
                            redirects,
                        ))
                    };

                    if let Some(callee_key) = callee_key.as_ref() {
                        if let Some(callee_spec) = specializations.get(callee_key) {
                            for (result, ret_wt) in results.iter().zip(&callee_spec.return_types) {
                                state.join_value(*result, ret_wt.clone());
                            }
                            for (arg, arg_wt) in args.iter().zip(&callee_key.arg_types) {
                                state.join_value(*arg, arg_wt.clone());
                            }
                            continue;
                        }
                    }

                    let callee_func = ssa.get_function(*callee_id);
                    for (result, ret_type) in results.iter().zip(callee_func.get_returns()) {
                        state.join_value(*result, Self::construct_pure_witness_for_type(ret_type));
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
                    let result_elem_wt = elems
                        .iter()
                        .fold(Self::construct_pure_witness_for_type(tp), |acc, elem| {
                            acc.join(state.witness(*elem))
                        });
                    state.join_value(
                        *result,
                        WitnessShape::Array(WitnessType::Pure, Box::new(result_elem_wt)),
                    );
                    let result_elem_wt = state.array_element(*result);
                    state.back_values(elems, &result_elem_wt);
                }
                OpCode::MkRepeated {
                    result,
                    element,
                    seq_type: _,
                    count: _,
                    elem_type: tp,
                } => {
                    let result_elem_wt =
                        Self::construct_pure_witness_for_type(tp).join(state.witness(*element));
                    state.join_value(
                        *result,
                        WitnessShape::Array(WitnessType::Pure, Box::new(result_elem_wt)),
                    );
                    let result_elem_wt = state.array_element(*result);
                    state.back_value(*element, &result_elem_wt);
                }
                OpCode::Unspread {
                    result_odd,
                    result_even,
                    value,
                    ..
                } => {
                    let val_wt = state.witness(*value).clone();
                    let odd_wt = state.join_value(*result_odd, val_wt.clone());
                    let even_wt = state.join_value(*result_even, val_wt);
                    state.back_value(*value, &odd_wt);
                    state.back_value(*value, &even_wt);
                }
                OpCode::Spread { result, value, .. }
                | OpCode::Cast { result, value, .. }
                | OpCode::SExt { result, value, .. }
                | OpCode::BitRange { result, value, .. }
                | OpCode::Not { result, value } => {
                    let result_wt = state.join_value(*result, state.witness(*value).clone());
                    state.back_value(*value, &result_wt);
                }
                OpCode::ToBits { result, value, .. } | OpCode::ToRadix { result, value, .. } => {
                    state.join_value(
                        *result,
                        WitnessShape::Array(
                            WitnessType::Pure,
                            Box::new(state.witness(*value).clone()),
                        ),
                    );
                    let result_elem_wt = state.array_element(*result);
                    state.back_value(*value, &result_elem_wt);
                }
                OpCode::TupleProj { result, tuple, idx } => {
                    let (tuple_top, elem_wt) = state.tuple_element(*tuple, *idx);
                    let result_wt =
                        elem_wt.with_toplevel_info(tuple_top.join(elem_wt.toplevel_info()));
                    let result_wt = state.join_value(*result, result_wt);
                    state.back_tuple_element(*tuple, *idx, &result_wt);
                }
                OpCode::MkTuple { result, elems, .. } => {
                    let children: Vec<WitnessShape> =
                        elems.iter().map(|v| state.witness(*v).clone()).collect();
                    state.join_value(*result, WitnessShape::Tuple(WitnessType::Pure, children));
                    let result_wt = state.witness(*result).clone();
                    if let WitnessShape::Tuple(_, children) = result_wt {
                        for (elem, child_wt) in elems.iter().zip(children.iter()) {
                            state.back_value(*elem, child_wt);
                        }
                    }
                }
                OpCode::WriteWitness { result, .. } => {
                    // WriteWitness records a value on the witness tape.
                    // Its output is always Witness-typed.
                    if let Some(result) = result {
                        state.join_value(*result, WitnessShape::Scalar(WitnessType::Witness));
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
