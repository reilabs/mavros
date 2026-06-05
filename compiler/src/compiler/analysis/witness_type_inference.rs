//! Performs whole program analysis to determine which values are potentially witness tainted, which
//! are _only_ witnesses, and which are only non-witness values.
//!
//! See `docs/WITNESS_TYPE_INFERENCE.md` for the algorithm-level description.

mod fixpoint;
mod signature;
mod solver;
mod specialization;

use std::collections::HashMap;

use self::{
    fixpoint::infer_summaries,
    signature::{BoundaryLayout, SpecKey, pure_shape_for_type},
    solver::SpecSolver,
    specialization::Specializer,
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

    /// Runs WTI as explicit phases: initialize boundary data, infer summaries, solve specs,
    /// materialize clones, then install annotations for the specialized SSA functions.
    pub fn run(&mut self, ssa: &mut HLSSA, flow_analysis: &FlowAnalysis) -> Result<(), String> {
        // Phase 1: initialize the fixed boundary layout for every original function.
        let function_ids = ssa.get_function_ids().collect::<Vec<_>>();
        let layouts = function_ids
            .iter()
            .map(|function_id| (*function_id, BoundaryLayout::new(*function_id, ssa)))
            .collect::<HashMap<_, _>>();

        // Phase 2: infer whole-program boundary summaries over the original call graph.
        let summaries = infer_summaries(ssa, flow_analysis, &function_ids, &layouts);

        // Phase 3: build the root key. `main` starts pure at the boundary; closure decides what
        // the program actually needs from there.
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

        // Phase 4: solve the closed specialization graph without mutating SSA.
        let solved = SpecSolver::new(&*ssa, flow_analysis, &layouts, &summaries).solve(main_key);

        // Phase 5: clone solved specs, rewrite calls, and emit annotations for cloned functions.
        self.functions = Specializer::new(ssa, flow_analysis, solved).materialize();

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
