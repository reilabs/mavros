//! Performs whole program analysis to determine which values are potentially witness tainted, which
//! are _only_ witnesses, and which are only non-witness values.
//!
//! See `docs/WITNESS_TYPE_INFERENCE.md` for the algorithm-level description.

mod fixpoint;
mod signature;
mod specialization;

use std::collections::HashMap;

use self::{
    fixpoint::infer_summaries,
    signature::{BoundaryLayout, SpecKey, pure_shape_for_type},
    specialization::SpecializationEngine,
};
use super::witness_info::{FunctionWitnessType, WitnessType};
use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    ssa::{BlockId, FunctionId, SSAAnotator, ValueId, hlssa::HLSSA},
};

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

    /// Runs summary inference, materializes the closed specialization graph, and installs the
    /// resulting witness annotations for the specialized SSA functions.
    pub fn run(&mut self, ssa: &mut HLSSA, flow_analysis: &FlowAnalysis) -> Result<(), String> {
        let function_ids = ssa.get_function_ids().collect::<Vec<_>>();
        let layouts = function_ids
            .iter()
            .map(|function_id| (*function_id, BoundaryLayout::new(*function_id, ssa)))
            .collect::<HashMap<_, _>>();
        let summaries = infer_summaries(ssa, flow_analysis, &function_ids, &layouts);

        let main_id = ssa.get_main_id();
        let main_func = ssa.get_function(main_id);
        let main_key = SpecKey {
            original_func_id: main_id,
            parameters: main_func
                .get_entry()
                .get_parameters()
                .map(|(_, tp)| pure_shape_for_type(tp))
                .collect(),
            returns: main_func
                .get_returns()
                .iter()
                .map(pure_shape_for_type)
                .collect(),
            cfg_witness: WitnessType::Pure,
        };

        let mut engine = SpecializationEngine::new(ssa, flow_analysis, layouts, summaries);
        let root_spec = engine.ensure_spec(main_key);
        engine.run();
        self.functions = engine.finish(root_spec);

        Ok(())
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
