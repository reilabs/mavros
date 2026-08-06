//! Validation for Noir's `assert_constant` compiler builtin.
//!
//! Lowering preserves each assertion as an [`OpCode::AssertConstant`] marker until witness-taint
//! inference has specialized functions for their concrete calling contexts. An assertion succeeds
//! exactly when its operand is `Pure` at every level in every specialized context at the point this
//! pass runs. Optimizations scheduled before validation can therefore affect which operands WTI
//! classifies as `Pure`.
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
        hlssa::{CallTarget, HLSSA, OpCode},
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
        let mut failures = BTreeSet::new();
        let mut functions_with_markers = Vec::new();

        for (fid, function) in ssa.iter_functions() {
            let witness_types = self.witness_inference.try_get_function_witness_type(*fid);
            let mut has_marker = false;

            for (_, block) in function.get_blocks() {
                for (op, location) in block.get_instructions_with_source_locations() {
                    assert!(
                        !matches!(op, OpCode::Guard { .. }),
                        "ICE: Guard reached assert_constant validation before witness taint inference"
                    );
                    if witness_types.is_some()
                        && let OpCode::Call {
                            function: CallTarget::Static(callee),
                            unconstrained: false,
                            ..
                        } = op
                    {
                        debug_assert!(
                            self.witness_inference
                                .try_get_function_witness_type(*callee)
                                .is_some(),
                            "ICE: WTI-covered function {fid:?} has a constrained static call to uncovered function {callee:?}"
                        );
                    }
                    let OpCode::AssertConstant { value } = op else {
                        continue;
                    };
                    has_marker = true;
                    if witness_types
                        .and_then(|types| types.try_get_value_witness_type(*value))
                        .is_some_and(|shape| shape.contains_witness())
                    {
                        failures.insert(location.clone());
                    }
                }
            }

            if has_marker {
                functions_with_markers.push(*fid);
            }
        }

        let failures: Vec<_> = failures.into_iter().collect();

        if failures.is_empty() {
            for fid in functions_with_markers {
                let function = ssa.get_function_mut(fid);
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
            FunctionId, SourceLocation, SourcePosition, Terminator, ValueId,
            hlssa::{CallTarget, CastTarget, Constant, HLSSA, OpCode, SequenceTargetType, Type},
        },
    };

    fn assert_constant(value: ValueId) -> crate::compiler::ssa::Located<OpCode> {
        OpCode::AssertConstant { value }.locate(SourceLocation::test())
    }

    fn assert_constant_at(value: ValueId, line: u64) -> crate::compiler::ssa::Located<OpCode> {
        OpCode::AssertConstant { value }.locate(SourceLocation::new(
            "assert_constant.nr",
            SourcePosition::new(line, 1),
            SourcePosition::new(line, 20),
        ))
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
        // Once witness-length slices are supported, this test must also cover a genuinely dynamic
        // slice length and require validation to reject it.
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

    #[test]
    fn rejects_a_direct_witness_and_preserves_markers_on_failure() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let dynamic = witness(&mut ssa, five);
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_instruction(assert_constant(dynamic));
        entry.set_terminator(Terminator::Return(vec![]));

        assert_eq!(validate(&mut ssa).unwrap_err().len(), 1);
        assert!(ssa.iter_functions().any(|(_, function)| {
            function.get_blocks().any(|(_, block)| {
                block
                    .get_instructions()
                    .any(|op| matches!(op, OpCode::AssertConstant { .. }))
            })
        }));
    }

    #[test]
    fn reports_every_failing_assertion() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let dynamic = witness(&mut ssa, five);
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_instruction(assert_constant_at(dynamic, 10));
        entry.push_instruction(assert_constant_at(dynamic, 20));
        entry.set_terminator(Terminator::Return(vec![]));

        let failures = validate(&mut ssa).unwrap_err();
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].start.line, 10);
        assert_eq!(failures[1].start.line, 20);
    }

    #[test]
    #[should_panic(
        expected = "ICE: Guard reached assert_constant validation before witness taint inference"
    )]
    fn rejects_a_guard_before_witness_taint_inference() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let five = ssa.add_const(Constant::Field(Field::from(5u64)));
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_instruction(
            OpCode::Guard {
                condition: five,
                inner: Box::new(OpCode::AssertConstant { value: five }),
            }
            .locate(SourceLocation::test()),
        );
        entry.set_terminator(Terminator::Return(vec![]));

        let _ = validate(&mut ssa);
    }

    #[test]
    #[should_panic(expected = "has a constrained static call to uncovered function")]
    fn detects_missing_wti_coverage_for_a_constrained_callee() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.get_unique_entrypoint_mut()
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        let flow = FlowAnalysis::run(&ssa);
        let mut witness_inference = WitnessTaintInference::new();
        witness_inference.run(&mut ssa, &flow);

        // Simulate a future WTI coverage bug by introducing a constrained callee after inference.
        let uncovered = ssa.add_function("uncovered".to_string());
        ssa.get_function_mut(uncovered)
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));
        let main = ssa.get_unique_entrypoint_id();
        ssa.get_function_mut(main).get_entry_mut().push_instruction(
            OpCode::Call {
                results: vec![],
                function: CallTarget::Static(uncovered),
                args: vec![],
                unconstrained: false,
            }
            .locate(SourceLocation::test()),
        );

        let failures = Rc::new(RefCell::new(None));
        PassManager::new(
            "assert_constant_validation_test".to_string(),
            false,
            vec![Box::new(AssertConstantValidation::new(
                Rc::new(witness_inference),
                failures,
            ))],
        )
        .run(&mut ssa);
    }
}
