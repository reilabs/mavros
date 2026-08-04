//! Validation for Noir's `assert_constant` compiler builtin.
//!
//! Lowering preserves each assertion as an [`OpCode::AssertConstant`] marker until witness-taint
//! inference has specialized functions for their concrete calling contexts. An assertion succeeds
//! exactly when its operand is `Pure` at every level in every specialized context.
//!
//! WTI intentionally omits unconstrained-only callees. Assertions that exist only in those
//! functions are therefore outside this validation; successful validation still removes their
//! markers so later passes never need to interpret the compiler-only opcode.

use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

use crate::compiler::{
    analysis::witness_taint_inference::WitnessTaintInference,
    pass_manager::{AnalysisStore, Pass},
    ssa::{
        SourceLocation,
        hlssa::{HLSSA, OpCode},
    },
};

/// Validate every `AssertConstant` against WTI and erase all markers when validation succeeds.
///
/// Constants are intentionally absent from the per-function value-shape map, so a missing value
/// shape is `Pure`. Aggregate shapes are checked recursively: a container is accepted only when
/// none of its levels contains a witness.
pub(crate) struct AssertConstantValidation {
    witness_inference: Rc<WitnessTaintInference>,
    failures: Rc<RefCell<Option<Vec<SourceLocation>>>>,
}

impl AssertConstantValidation {
    pub(crate) fn new(
        witness_inference: Rc<WitnessTaintInference>,
        failures: Rc<RefCell<Option<Vec<SourceLocation>>>>,
    ) -> Self {
        Self {
            witness_inference,
            failures,
        }
    }
}

impl Pass for AssertConstantValidation {
    fn name(&self) -> &'static str {
        "assert_constant_validation"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        let failures: Vec<SourceLocation> = ssa
            .iter_functions()
            .filter_map(|(fid, function)| {
                self.witness_inference
                    .try_get_function_witness_type(*fid)
                    .map(|witness_types| (function, witness_types))
            })
            .flat_map(|(function, witness_types)| {
                function.get_blocks().flat_map(move |(_, block)| {
                    block.get_instructions_with_source_locations().filter_map(
                        move |(op, location)| {
                            let OpCode::AssertConstant { value } = op else {
                                return None;
                            };
                            witness_types
                                .try_get_value_witness_type(*value)
                                .is_some_and(|shape| shape.contains_witness())
                                .then(|| location.clone())
                        },
                    )
                })
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        if failures.is_empty() {
            for (_, function) in ssa.iter_functions_mut() {
                for (_, block) in function.get_blocks_mut() {
                    let instructions = block.take_instructions();
                    block.put_instructions(
                        instructions
                            .into_iter()
                            .filter(|op| !matches!(&**op, OpCode::AssertConstant { .. }))
                            .collect(),
                    );
                }
            }
        }

        *self.failures.borrow_mut() = Some(failures);
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::AssertConstantValidation;
    use crate::compiler::{
        Field,
        analysis::{flow_analysis::FlowAnalysis, witness_taint_inference::WitnessTaintInference},
        pass_manager::PassManager,
        ssa::{
            FunctionId, SourceLocation, Terminator, ValueId,
            hlssa::{CallTarget, CastTarget, Constant, HLSSA, OpCode, SequenceTargetType, Type},
        },
    };

    fn assert_constant(value: ValueId) -> crate::compiler::ssa::Located<OpCode> {
        OpCode::AssertConstant { value }.locate(SourceLocation::test())
    }

    fn validate(ssa: &mut HLSSA) -> Result<(), Vec<SourceLocation>> {
        let flow = FlowAnalysis::run(ssa);
        let mut witness_inference = WitnessTaintInference::new();
        witness_inference.run(ssa, &flow);
        let witness_inference = Rc::new(witness_inference);
        let failures = Rc::new(RefCell::new(None));
        PassManager::new(
            "assert_constant_validation_test".to_string(),
            false,
            vec![Box::new(AssertConstantValidation::new(
                witness_inference,
                Rc::clone(&failures),
            ))],
        )
        .run(ssa);
        let failures = failures.borrow_mut().take().unwrap();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }

    fn add_asserting_helper(ssa: &mut HLSSA) -> FunctionId {
        let helper = ssa.add_function("asserting_helper".to_string());
        let parameter = ssa.fresh_value();
        let entry = ssa.get_function_mut(helper).get_entry_mut();
        entry.push_parameter(parameter, Type::field());
        entry.push_instruction(assert_constant(parameter));
        entry.set_terminator(Terminator::Return(vec![]));
        helper
    }

    fn call(ssa: &mut HLSSA, callee: FunctionId, argument: ValueId) {
        let main = ssa.get_unique_entrypoint_id();
        ssa.get_function_mut(main).get_entry_mut().push_instruction(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(callee),
                args: vec![argument],
                unconstrained: false,
            }
            .locate(SourceLocation::test()),
        );
    }

    fn witness(ssa: &mut HLSSA, value: ValueId) -> ValueId {
        let result = ssa.fresh_value();
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .push_instruction(
                OpCode::WriteWitness {
                    result: Some(result),
                    value,
                    pinned: false,
                }
                .locate(SourceLocation::test()),
            );
        result
    }

    #[test]
    fn accepts_constants_in_every_call_context_and_removes_markers() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let helper = add_asserting_helper(&mut ssa);
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let six = ssa.add_const(Constant::Field(Field::from(6u64)));
        call(&mut ssa, helper, five);
        call(&mut ssa, helper, six);
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        validate(&mut ssa).unwrap();
        assert!(ssa.iter_functions().all(|(_, function)| {
            function.get_blocks().all(|(_, block)| {
                block
                    .get_instructions()
                    .all(|op| !matches!(op, OpCode::AssertConstant { .. }))
            })
        }));
    }

    #[test]
    fn rejects_when_any_call_context_contains_a_witness() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let helper = add_asserting_helper(&mut ssa);
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let dynamic = witness(&mut ssa, five);
        call(&mut ssa, helper, five);
        call(&mut ssa, helper, dynamic);
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        assert_eq!(validate(&mut ssa).unwrap_err().len(), 1);
    }

    #[test]
    fn recursively_checks_aggregate_shapes() {
        for dynamic_element in [false, true] {
            let mut ssa = HLSSA::with_main("main".to_string());
            let five = ssa.add_const(Constant::Field(Field::from(5u64)));
            let element = if dynamic_element {
                witness(&mut ssa, five)
            } else {
                five
            };
            let array = ssa.fresh_value();
            let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
            entry.push_instruction(
                OpCode::MkSeq {
                    result: array,
                    elems: vec![element],
                    seq_type: SequenceTargetType::Array(1),
                    elem_type: Type::field(),
                }
                .locate(SourceLocation::test()),
            );
            entry.push_instruction(assert_constant(array));
            entry.set_terminator(Terminator::Return(vec![]));

            assert_eq!(validate(&mut ssa).is_ok(), !dynamic_element);
        }
    }

    #[test]
    fn accepts_static_sequence_length_with_witness_elements() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let dynamic = witness(&mut ssa, five);
        let array = ssa.fresh_value();
        let slice = ssa.fresh_value();
        let len = ssa.fresh_value();
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_instruction(
            OpCode::MkRepeated {
                result: array,
                element: dynamic,
                seq_type: SequenceTargetType::Array(17),
                count: 17,
                elem_type: Type::field(),
            }
            .locate(SourceLocation::test()),
        );
        entry.push_instruction(
            OpCode::Cast {
                result: slice,
                value: array,
                target: CastTarget::ArrayToSlice,
            }
            .locate(SourceLocation::test()),
        );
        entry.push_instruction(
            OpCode::SliceLen { result: len, slice }.locate(SourceLocation::test()),
        );
        entry.push_instruction(assert_constant(len));
        entry.set_terminator(Terminator::Return(vec![]));

        validate(&mut ssa).unwrap();
    }
}
