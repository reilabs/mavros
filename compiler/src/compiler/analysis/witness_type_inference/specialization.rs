use std::collections::{HashMap, HashSet, VecDeque};

use super::super::witness_info::{FunctionWitnessType, WitnessShape, WitnessType};
use super::{
    fixpoint::BodyBuilder,
    signature::{
        BoundaryLayout, CallSite, FunctionSummary, PendingCall, SpecKey, VarShape, VariableId,
        collect_witness_ports, concrete_shape_from_ports, concretize_shape, witness_of_var,
    },
};
use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    ssa::{
        BlockId, FunctionId, Instruction, Terminator, ValueId,
        hlssa::{CallTarget, HLSSA, OpCode},
    },
};

#[derive(Clone, Debug)]
struct SpecInstance {
    original_func_id: FunctionId,
    specialized_func_id: FunctionId,
    key: SpecKey,
    data: Option<SpecData>,
}

#[derive(Clone, Debug)]
struct SpecData {
    entry_params: Vec<VarShape>,
    return_shapes: Vec<VarShape>,
    cfg_var: VariableId,
    value_shapes: HashMap<ValueId, VarShape>,
    block_cfg_vars: HashMap<BlockId, VariableId>,
    witness_vars: HashSet<VariableId>,
}

/// Materializes the closed specialization graph and rewrites calls to the selected clones.
pub(super) struct SpecializationEngine<'a> {
    ssa: &'a mut HLSSA,
    flow_analysis: &'a FlowAnalysis,
    layouts: HashMap<FunctionId, BoundaryLayout>,
    summaries: HashMap<FunctionId, FunctionSummary>,
    specs: Vec<SpecInstance>,
    specs_by_key: HashMap<SpecKey, usize>,
    spec_queue: VecDeque<usize>,
    resolved_calls: HashMap<CallSite, usize>,
}

impl<'a> SpecializationEngine<'a> {
    pub(super) fn new(
        ssa: &'a mut HLSSA,
        flow_analysis: &'a FlowAnalysis,
        layouts: HashMap<FunctionId, BoundaryLayout>,
        summaries: HashMap<FunctionId, FunctionSummary>,
    ) -> Self {
        Self {
            ssa,
            flow_analysis,
            layouts,
            summaries,
            specs: Vec::new(),
            specs_by_key: HashMap::new(),
            spec_queue: VecDeque::new(),
            resolved_calls: HashMap::new(),
        }
    }

    pub(super) fn run(&mut self) {
        while let Some(spec_idx) = self.spec_queue.pop_front() {
            self.scan_spec(spec_idx);
        }
    }

    /// Returns the canonical specialization for `key`, after closing it under the function summary.
    pub(super) fn ensure_spec(&mut self, key: SpecKey) -> usize {
        let key = self.close_key(key);
        if let Some(idx) = self.specs_by_key.get(&key) {
            return *idx;
        }

        let specialized_func_id = self.ssa.duplicate_function(key.original_func_id);
        let spec_idx = self.specs.len();
        self.specs.push(SpecInstance {
            original_func_id: key.original_func_id,
            specialized_func_id,
            key: key.clone(),
            data: None,
        });
        self.specs_by_key.insert(key, spec_idx);
        self.spec_queue.push_back(spec_idx);
        spec_idx
    }

    /// Builds one specialization's value graph, records its concrete witness facts, and queues
    /// closed callees for all constrained static calls found in its body.
    fn scan_spec(&mut self, spec_idx: usize) {
        if self.specs[spec_idx].data.is_some() {
            return;
        }

        let spec = self.specs[spec_idx].clone();
        let build = BodyBuilder::for_spec(
            &spec.key,
            spec.specialized_func_id,
            &*self.ssa,
            self.flow_analysis,
            &self.layouts,
            &self.summaries,
        )
        .build();
        let witness_vars = build.graph.reachable_from([build.always]);
        let data = SpecData {
            entry_params: build.entry_params,
            return_shapes: build.return_shapes,
            cfg_var: build.cfg_var,
            value_shapes: build.value_shapes,
            block_cfg_vars: build.block_cfg_vars,
            witness_vars,
        };

        let actual_key = self.key_from_spec_data(&spec, &data);
        assert_eq!(
            self.close_key(actual_key),
            spec.key,
            "Specialization key was not closed for {:?}",
            spec.key
        );

        for call in build.pending_calls {
            let callee_key = self.key_from_call(&data, &call);
            let callee_spec = self.ensure_spec(callee_key);
            self.resolved_calls.insert(call.call_site, callee_spec);
        }

        self.specs[spec_idx].data = Some(data);
    }

    /// Applies all call-target rewrites and returns the witness annotations for materialized specs.
    pub(super) fn finish(mut self, root_spec: usize) -> HashMap<FunctionId, FunctionWitnessType> {
        let spec_indices = (0..self.specs.len()).collect::<Vec<_>>();
        for spec_idx in spec_indices {
            self.rewrite_calls(spec_idx);
        }

        let root_func = self.specs[root_spec].specialized_func_id;
        self.ssa.set_entry_point(root_func);

        let mut functions = HashMap::new();
        for spec in &self.specs {
            let data = spec
                .data
                .as_ref()
                .unwrap_or_else(|| panic!("Unscanned witness specialization {:?}", spec.key));
            let function_id = spec.specialized_func_id;
            let function = self.ssa.get_function(function_id);
            let block_order = self
                .flow_analysis
                .get_function_cfg(spec.original_func_id)
                .get_blocks_bfs()
                .collect::<Vec<_>>();

            let mut block_cfg_witness = HashMap::new();
            let mut value_witness_types = HashMap::new();
            for block_id in block_order {
                let cfg_var = data.block_cfg_vars[&block_id];
                block_cfg_witness.insert(block_id, witness_of_var(cfg_var, &data.witness_vars));

                let block = function.get_block(block_id);
                for (value, _) in block.get_parameters() {
                    insert_concrete_value_shape(*value, data, &mut value_witness_types);
                }
                for instruction in block.get_instructions() {
                    for value in instruction.get_inputs().chain(instruction.get_results()) {
                        insert_concrete_value_shape(*value, data, &mut value_witness_types);
                    }
                }
                if let Some(terminator) = block.get_terminator() {
                    match terminator {
                        Terminator::Jmp(_, values) | Terminator::Return(values) => {
                            for value in values {
                                insert_concrete_value_shape(*value, data, &mut value_witness_types);
                            }
                        }
                        Terminator::JmpIf(cond, _, _) => {
                            insert_concrete_value_shape(*cond, data, &mut value_witness_types);
                        }
                    }
                }
            }

            functions.insert(
                function_id,
                FunctionWitnessType {
                    returns_witness: data
                        .return_shapes
                        .iter()
                        .map(|shape| concretize_shape(shape, &data.witness_vars))
                        .collect(),
                    cfg_witness: witness_of_var(data.cfg_var, &data.witness_vars),
                    parameters: data
                        .entry_params
                        .iter()
                        .map(|shape| concretize_shape(shape, &data.witness_vars))
                        .collect(),
                    block_cfg_witness,
                    value_witness_types,
                },
            );
        }

        functions
    }

    /// Swaps each constrained static call to the specialized callee selected during scanning.
    fn rewrite_calls(&mut self, spec_idx: usize) {
        let spec = self.specs[spec_idx].clone();
        let block_order = self
            .flow_analysis
            .get_function_cfg(spec.original_func_id)
            .get_blocks_bfs()
            .collect::<Vec<_>>();
        let mut rewrites = HashMap::<(BlockId, usize), FunctionId>::new();

        for block_id in block_order {
            let block = self
                .ssa
                .get_function(spec.specialized_func_id)
                .get_block(block_id);
            for (idx, instruction) in block.get_instructions().enumerate() {
                let OpCode::Call {
                    function: CallTarget::Static(_),
                    unconstrained,
                    ..
                } = instruction
                else {
                    continue;
                };
                if *unconstrained {
                    continue;
                }

                let call_site = CallSite {
                    caller_func_id: spec.specialized_func_id,
                    block_id,
                    instruction_idx: idx,
                };
                let callee_spec = self
                    .resolved_calls
                    .get(&call_site)
                    .unwrap_or_else(|| panic!("Unresolved witness call site {:?}", call_site));
                rewrites.insert(
                    (block_id, idx),
                    self.specs[*callee_spec].specialized_func_id,
                );
            }
        }

        let function = self.ssa.get_function_mut(spec.specialized_func_id);
        for ((block_id, idx), callee_id) in rewrites {
            let instruction = function
                .get_block_mut(block_id)
                .get_instructions_mut()
                .nth(idx)
                .unwrap_or_else(|| {
                    panic!("Missing call instruction {} in block {:?}", idx, block_id)
                });
            let OpCode::Call { function, .. } = instruction else {
                panic!("Call rewrite target is no longer a call");
            };
            let CallTarget::Static(target) = function else {
                panic!("Call rewrite target is no longer a static call");
            };
            *target = callee_id;
        }
    }

    /// Normalizes an incoming key by applying the callee's boundary summary to fixed point.
    fn close_key(&self, key: SpecKey) -> SpecKey {
        let layout = &self.layouts[&key.original_func_id];
        let summary = &self.summaries[&key.original_func_id];
        let mut active = HashSet::new();

        for (shape, concrete) in layout.parameters.iter().zip(&key.parameters) {
            collect_witness_ports(shape, concrete, &mut active);
        }
        for (shape, concrete) in layout.returns.iter().zip(&key.returns) {
            collect_witness_ports(shape, concrete, &mut active);
        }
        if key.cfg_witness.is_witness() {
            active.insert(layout.cfg.0);
        }

        let closed = summary.close(layout, active);
        SpecKey {
            original_func_id: key.original_func_id,
            parameters: layout
                .parameters
                .iter()
                .map(|shape| concrete_shape_from_ports(shape, &closed))
                .collect(),
            returns: layout
                .returns
                .iter()
                .map(|shape| concrete_shape_from_ports(shape, &closed))
                .collect(),
            cfg_witness: if closed.contains(&layout.cfg.0) {
                WitnessType::Witness
            } else {
                WitnessType::Pure
            },
        }
    }

    fn key_from_spec_data(&self, spec: &SpecInstance, data: &SpecData) -> SpecKey {
        SpecKey {
            original_func_id: spec.original_func_id,
            parameters: data
                .entry_params
                .iter()
                .map(|shape| concretize_shape(shape, &data.witness_vars))
                .collect(),
            returns: data
                .return_shapes
                .iter()
                .map(|shape| concretize_shape(shape, &data.witness_vars))
                .collect(),
            cfg_witness: witness_of_var(data.cfg_var, &data.witness_vars),
        }
    }

    fn key_from_call(&self, data: &SpecData, call: &PendingCall) -> SpecKey {
        SpecKey {
            original_func_id: call.callee_id,
            parameters: call
                .args
                .iter()
                .map(|value| concretize_shape(&data.value_shapes[value], &data.witness_vars))
                .collect(),
            returns: call
                .results
                .iter()
                .map(|value| concretize_shape(&data.value_shapes[value], &data.witness_vars))
                .collect(),
            cfg_witness: witness_of_var(call.cfg_var, &data.witness_vars),
        }
    }
}

fn insert_concrete_value_shape(
    value: ValueId,
    data: &SpecData,
    output: &mut HashMap<ValueId, WitnessShape>,
) {
    if let Some(shape) = data.value_shapes.get(&value) {
        output.insert(value, concretize_shape(shape, &data.witness_vars));
    }
}
