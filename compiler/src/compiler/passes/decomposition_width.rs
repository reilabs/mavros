//! Validation of the witnessed bit and byte decompositions against the configured field.
//!
//! A witnessed `to_bits` or `to_radix` wider than one field element is refused by its lowering,
//! but runs inside a pipeline that cannot return an error.

use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::Types},
    diagnostic::Diagnostic,
    pass_manager::{AnalysisStore, Pass},
    passes::{
        instruction_lowering::{witness_to_bits_refusal, witness_to_radix_refusal},
        shared::unsupported::field_offers,
    },
    ssa::hlssa::{HLSSA, OpCode},
};

/// Record a diagnostic for every witnessed decomposition the configured field cannot carry.
///
/// It must run immediately before `InstructionLowering::witness_integer_ops`, whose
/// `LowerWitnessFieldOps` lowers these decompositions, so the two see the same instructions and
/// types. It changes nothing.
pub(crate) struct DecompositionWidthValidation {
    refusals: Rc<RefCell<Option<Vec<Diagnostic>>>>,
}

impl DecompositionWidthValidation {
    pub(crate) fn new(refusals: Rc<RefCell<Option<Vec<Diagnostic>>>>) -> Self {
        Self { refusals }
    }
}

impl Pass for DecompositionWidthValidation {
    fn name(&self) -> &'static str {
        "decomposition_width_validation"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        let field = ssa.field();
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);

        // One refusal per source location, however many specializations or unrolled copies of the
        // instruction there are.
        let mut refusals = BTreeSet::new();
        for (function_id, function) in ssa.iter_functions() {
            let type_info = types.get_function(*function_id);
            for (_, block) in function.get_blocks() {
                for (op, location) in block.get_instructions_with_source_locations() {
                    let op = match op {
                        OpCode::Guard { inner, .. } => inner.as_ref(),
                        op => op,
                    };
                    let (value, refusal) = match op {
                        OpCode::ToBits { value, count, .. } => {
                            (value, witness_to_bits_refusal(*count, field))
                        }
                        OpCode::ToRadix { value, count, .. } => {
                            (value, witness_to_radix_refusal(*count, field))
                        }
                        _ => continue,
                    };
                    if let Some(refusal) = refusal
                        && type_info.get_value_type(*value).is_witness_of()
                    {
                        refusals.insert((location.clone(), refusal));
                    }
                }
            }
        }

        *self.refusals.borrow_mut() = Some(
            refusals
                .into_iter()
                .map(|(location, refusal)| {
                    Diagnostic::error(refusal, location).with_note(field_offers(field))
                })
                .collect(),
        );
    }
}
