//! Reject entry points with no returning path before optimizations can erase their calls.

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, return_reachability::non_returning_entries},
    pass_manager::{AnalysisId, AnalysisStore, Pass, PassFailure, PassResult},
    ssa::hlssa::HLSSA,
};

pub(crate) struct ReturnReachabilityValidation;

impl ReturnReachabilityValidation {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Pass for ReturnReachabilityValidation {
    fn name(&self) -> &'static str {
        "return_reachability_validation"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) -> PassResult {
        let failures = non_returning_entries(ssa, store.get::<FlowAnalysis>());
        if failures.is_empty() {
            Ok(())
        } else {
            Err(PassFailure::Unsatisfiable(failures))
        }
    }
}
