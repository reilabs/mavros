//! Reject entry points with no returning path before optimizations can erase their calls.

use std::{cell::RefCell, rc::Rc};

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, return_reachability::non_returning_entries},
    diagnostic::Diagnostic,
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    ssa::hlssa::HLSSA,
};

pub(crate) struct ReturnReachabilityValidation {
    failures: Rc<RefCell<Option<Vec<Diagnostic>>>>,
}

impl ReturnReachabilityValidation {
    pub(crate) fn new(failures: Rc<RefCell<Option<Vec<Diagnostic>>>>) -> Self {
        Self { failures }
    }
}

impl Pass for ReturnReachabilityValidation {
    fn name(&self) -> &'static str {
        "return_reachability_validation"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        *self.failures.borrow_mut() = Some(non_returning_entries(ssa, store.get::<FlowAnalysis>()));
    }
}
