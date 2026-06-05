use std::collections::{HashMap, HashSet, VecDeque};

use super::super::witness_info::WitnessType;
use super::fixpoint::BodyBuilder;
use super::signature::{
    BoundaryLayout, CallSite, FunctionSummary, SpecKey, VarShape, VariableId,
    collect_witness_ports, concrete_shape_from_ports, concretize_shape, witness_of_var,
};
use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    ssa::{
        BlockId, FunctionId, ValueId,
        hlssa::{CallTarget, HLSSA, OpCode},
    },
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SpecId(pub(super) usize);

#[derive(Clone, Debug)]
struct SpecSlot {
    original_func_id: FunctionId,
    key: SpecKey,
    data: Option<SpecData>,
}

#[derive(Clone, Debug)]
pub(super) struct SpecData {
    pub(super) entry_params: Vec<VarShape>,
    pub(super) return_shapes: Vec<VarShape>,
    pub(super) cfg_var: VariableId,
    pub(super) value_shapes: HashMap<ValueId, VarShape>,
    pub(super) block_cfg_vars: HashMap<BlockId, VariableId>,
    pub(super) witness_vars: HashSet<VariableId>,
}

#[derive(Clone, Debug)]
pub(super) struct SolvedSpec {
    pub(super) original_func_id: FunctionId,
    pub(super) key: SpecKey,
    pub(super) data: SpecData,
}

#[derive(Clone, Debug)]
pub(super) struct SolvedProgram {
    pub(super) root: SpecId,
    pub(super) specs: Vec<SolvedSpec>,
    pub(super) resolved_calls: HashMap<(SpecId, CallSite), SpecId>,
}

/// Solves closed specialization keys over original functions only.
pub(super) struct SpecSolver<'a> {
    ssa: &'a HLSSA,
    flow_analysis: &'a FlowAnalysis,
    layouts: &'a HashMap<FunctionId, BoundaryLayout>,
    summaries: &'a HashMap<FunctionId, FunctionSummary>,
    specs: Vec<SpecSlot>,
    specs_by_key: HashMap<SpecKey, SpecId>,
    spec_queue: VecDeque<SpecId>,
    resolved_calls: HashMap<(SpecId, CallSite), SpecId>,
}

impl<'a> SpecSolver<'a> {
    pub(super) fn new(
        ssa: &'a HLSSA,
        flow_analysis: &'a FlowAnalysis,
        layouts: &'a HashMap<FunctionId, BoundaryLayout>,
        summaries: &'a HashMap<FunctionId, FunctionSummary>,
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

    /// Runs the solve queue to completion and returns immutable solved results.
    pub(super) fn solve(mut self, root_key: SpecKey) -> SolvedProgram {
        let root = self.ensure_spec(root_key);
        while let Some(spec_id) = self.spec_queue.pop_front() {
            self.solve_spec(spec_id);
        }

        let specs = self
            .specs
            .into_iter()
            .map(|slot| SolvedSpec {
                original_func_id: slot.original_func_id,
                key: slot.key,
                data: slot
                    .data
                    .unwrap_or_else(|| panic!("Unsolved witness specialization")),
            })
            .collect();

        SolvedProgram {
            root,
            specs,
            resolved_calls: self.resolved_calls,
        }
    }

    /// Returns the canonical solved-spec slot for `key`, after closing it under the summary.
    fn ensure_spec(&mut self, key: SpecKey) -> SpecId {
        let key = self.close_key(key);
        if let Some(id) = self.specs_by_key.get(&key) {
            return *id;
        }

        let spec_id = SpecId(self.specs.len());
        self.specs.push(SpecSlot {
            original_func_id: key.original_func_id,
            key: key.clone(),
            data: None,
        });
        self.specs_by_key.insert(key, spec_id);
        self.spec_queue.push_back(spec_id);
        spec_id
    }

    /// Solves one closed key over the original function body and queues closed callee keys.
    fn solve_spec(&mut self, spec_id: SpecId) {
        if self.specs[spec_id.0].data.is_some() {
            return;
        }

        let slot = self.specs[spec_id.0].clone();
        let build = BodyBuilder::for_spec(
            &slot.key,
            self.ssa,
            self.flow_analysis,
            self.layouts,
            self.summaries,
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

        let actual_key = self.key_from_spec_data(&slot, &data);
        assert_eq!(
            self.close_key(actual_key),
            slot.key,
            "Specialization key was not closed for {:?}",
            slot.key
        );

        for call_site in build.constrained_calls {
            let callee_key = self.key_from_call_site(&data, call_site);
            let callee_spec = self.ensure_spec(callee_key);
            self.resolved_calls
                .insert((spec_id, call_site), callee_spec);
        }

        self.specs[spec_id.0].data = Some(data);
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

    fn key_from_spec_data(&self, slot: &SpecSlot, data: &SpecData) -> SpecKey {
        SpecKey {
            original_func_id: slot.original_func_id,
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

    fn key_from_call_site(&self, data: &SpecData, call_site: CallSite) -> SpecKey {
        let instruction = self
            .ssa
            .get_function(call_site.caller_func_id)
            .get_block(call_site.block_id)
            .get_instruction(call_site.instruction_idx);
        let OpCode::Call {
            results,
            function: CallTarget::Static(callee_id),
            args,
            unconstrained: false,
        } = instruction
        else {
            panic!("Witness call site is no longer a constrained static call");
        };
        let cfg_var = data.block_cfg_vars[&call_site.block_id];

        SpecKey {
            original_func_id: *callee_id,
            parameters: args
                .iter()
                .map(|value| concretize_shape(&data.value_shapes[value], &data.witness_vars))
                .collect(),
            returns: results
                .iter()
                .map(|value| concretize_shape(&data.value_shapes[value], &data.witness_vars))
                .collect(),
            cfg_witness: witness_of_var(cfg_var, &data.witness_vars),
        }
    }
}
