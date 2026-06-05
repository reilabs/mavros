use std::collections::HashMap;

use super::super::witness_info::{FunctionWitnessType, WitnessShape};
use super::signature::{CallSite, concretize_shape, witness_of_var};
use super::solver::{SolvedProgram, SpecData, SpecId};
use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    ssa::{
        BlockId, FunctionId, Instruction, Terminator, ValueId,
        hlssa::{CallTarget, HLSSA, OpCode},
    },
};

#[derive(Clone, Debug)]
struct MaterializedSpec {
    function_id: FunctionId,
    value_remap: HashMap<ValueId, ValueId>,
}

/// Materializes solved WTI specs by cloning functions, rewriting calls, and emitting annotations.
pub(super) struct Specializer<'a> {
    ssa: &'a mut HLSSA,
    flow_analysis: &'a FlowAnalysis,
    solved: SolvedProgram,
}

impl<'a> Specializer<'a> {
    pub(super) fn new(
        ssa: &'a mut HLSSA,
        flow_analysis: &'a FlowAnalysis,
        solved: SolvedProgram,
    ) -> Self {
        Self {
            ssa,
            flow_analysis,
            solved,
        }
    }

    /// Runs the materialization phase after solving is complete.
    pub(super) fn materialize(mut self) -> HashMap<FunctionId, FunctionWitnessType> {
        let materialized = self.clone_specs();
        self.rewrite_calls(&materialized);

        let root_func = materialized_spec(&materialized, self.solved.root).function_id;
        self.ssa.set_entry_point(root_func);

        self.collect_annotations(&materialized)
    }

    /// Creates one clone per solved spec and records the original-to-clone value remap.
    fn clone_specs(&mut self) -> Vec<MaterializedSpec> {
        let mut materialized = Vec::with_capacity(self.solved.specs.len());
        for spec_id in self.spec_ids() {
            let original_func_id = self.solved.specs[spec_id.0].original_func_id;
            let (function_id, value_remap) = self.ssa.duplicate_function(original_func_id);
            materialized.push(MaterializedSpec {
                function_id,
                value_remap,
            });
        }
        materialized
    }

    /// Applies the call graph selected during solving to the cloned functions.
    fn rewrite_calls(&mut self, materialized: &[MaterializedSpec]) {
        for spec_id in self.spec_ids() {
            self.rewrite_spec_calls(spec_id, materialized);
        }
    }

    fn rewrite_spec_calls(&mut self, spec_id: SpecId, materialized: &[MaterializedSpec]) {
        let spec = &self.solved.specs[spec_id.0];
        let function_id = materialized_spec(materialized, spec_id).function_id;
        let block_order = self
            .flow_analysis
            .get_function_cfg(spec.original_func_id)
            .get_blocks_bfs()
            .collect::<Vec<_>>();
        let mut rewrites = HashMap::<(BlockId, usize), FunctionId>::new();

        for block_id in block_order {
            let block = self
                .ssa
                .get_function(spec.original_func_id)
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
                    caller_func_id: spec.original_func_id,
                    block_id,
                    instruction_idx: idx,
                };
                let callee_spec = self
                    .solved
                    .resolved_calls
                    .get(&(spec_id, call_site))
                    .unwrap_or_else(|| {
                        panic!(
                            "Unresolved witness call site {:?} in spec {:?}",
                            call_site, spec.key
                        )
                    });
                rewrites.insert(
                    (block_id, idx),
                    materialized_spec(materialized, *callee_spec).function_id,
                );
            }
        }

        let function = self.ssa.get_function_mut(function_id);
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

    /// Builds the final annotation map for the cloned SSA functions.
    fn collect_annotations(
        &self,
        materialized: &[MaterializedSpec],
    ) -> HashMap<FunctionId, FunctionWitnessType> {
        let mut functions = HashMap::new();

        for spec_id in self.spec_ids() {
            let spec = &self.solved.specs[spec_id.0];
            let materialized = materialized_spec(materialized, spec_id);
            let original_function = self.ssa.get_function(spec.original_func_id);
            let block_order = self
                .flow_analysis
                .get_function_cfg(spec.original_func_id)
                .get_blocks_bfs()
                .collect::<Vec<_>>();

            let mut block_cfg_witness = HashMap::new();
            let mut value_witness_types = HashMap::new();
            for block_id in block_order {
                let cfg_var = spec.data.block_cfg_vars[&block_id];
                block_cfg_witness
                    .insert(block_id, witness_of_var(cfg_var, &spec.data.witness_vars));

                let block = original_function.get_block(block_id);
                for (value, _) in block.get_parameters() {
                    insert_concrete_value_shape(
                        *value,
                        remap_value(*value, &materialized.value_remap),
                        &spec.data,
                        &mut value_witness_types,
                    );
                }
                for instruction in block.get_instructions() {
                    for value in instruction.get_inputs().chain(instruction.get_results()) {
                        insert_concrete_value_shape(
                            *value,
                            remap_value(*value, &materialized.value_remap),
                            &spec.data,
                            &mut value_witness_types,
                        );
                    }
                }
                if let Some(terminator) = block.get_terminator() {
                    match terminator {
                        Terminator::Jmp(_, values) | Terminator::Return(values) => {
                            for value in values {
                                insert_concrete_value_shape(
                                    *value,
                                    remap_value(*value, &materialized.value_remap),
                                    &spec.data,
                                    &mut value_witness_types,
                                );
                            }
                        }
                        Terminator::JmpIf(cond, _, _) => {
                            insert_concrete_value_shape(
                                *cond,
                                remap_value(*cond, &materialized.value_remap),
                                &spec.data,
                                &mut value_witness_types,
                            );
                        }
                    }
                }
            }

            functions.insert(
                materialized.function_id,
                FunctionWitnessType {
                    returns_witness: spec
                        .data
                        .return_shapes
                        .iter()
                        .map(|shape| concretize_shape(shape, &spec.data.witness_vars))
                        .collect(),
                    cfg_witness: witness_of_var(spec.data.cfg_var, &spec.data.witness_vars),
                    parameters: spec
                        .data
                        .entry_params
                        .iter()
                        .map(|shape| concretize_shape(shape, &spec.data.witness_vars))
                        .collect(),
                    block_cfg_witness,
                    value_witness_types,
                },
            );
        }

        functions
    }

    fn spec_ids(&self) -> Vec<SpecId> {
        (0..self.solved.specs.len()).map(SpecId).collect()
    }
}

fn materialized_spec(materialized: &[MaterializedSpec], spec_id: SpecId) -> &MaterializedSpec {
    materialized
        .get(spec_id.0)
        .unwrap_or_else(|| panic!("Unmaterialized witness specialization {:?}", spec_id))
}

fn insert_concrete_value_shape(
    original_value: ValueId,
    output_value: ValueId,
    data: &SpecData,
    output: &mut HashMap<ValueId, WitnessShape>,
) {
    if let Some(shape) = data.value_shapes.get(&original_value) {
        output.insert(output_value, concretize_shape(shape, &data.witness_vars));
    }
}

fn remap_value(value: ValueId, remap: &HashMap<ValueId, ValueId>) -> ValueId {
    remap.get(&value).copied().unwrap_or(value)
}
