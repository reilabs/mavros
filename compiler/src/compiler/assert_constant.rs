//! Validation for Noir's `assert_constant` compiler builtin.
//!
//! Lowering preserves each assertion as an [`OpCode::AssertConstant`] marker until witness-taint
//! inference has specialized functions for their concrete calling contexts. An assertion succeeds
//! exactly when its operand is `Pure` at every level in every specialized context.

use std::collections::BTreeSet;

use crate::compiler::{
    analysis::witness_taint_inference::WitnessTaintInference,
    ssa::{
        SourceLocation,
        hlssa::{HLSSA, OpCode},
    },
};

/// Validate every specialized `AssertConstant` and erase all successfully validated markers.
///
/// Constants are intentionally absent from the per-function value-shape map, so a missing value
/// shape is `Pure`. Aggregate shapes are checked recursively: a container is constant only when
/// none of its levels contains a witness.
pub(crate) fn validate_and_remove(
    ssa: &mut HLSSA,
    witness_inference: &WitnessTaintInference,
) -> Result<(), Vec<SourceLocation>> {
    let failures: BTreeSet<SourceLocation> = ssa
        .iter_functions()
        .filter_map(|(fid, function)| {
            witness_inference
                .try_get_function_witness_type(*fid)
                .map(|witness_types| (function, witness_types))
        })
        .flat_map(|(function, witness_types)| {
            function.get_blocks().flat_map(move |(_, block)| {
                block
                    .get_instructions_with_source_locations()
                    .filter_map(move |(op, location)| {
                        let OpCode::AssertConstant { value } = op else {
                            return None;
                        };
                        witness_types
                            .try_get_value_witness_type(*value)
                            .is_some_and(|shape| shape.contains_witness())
                            .then(|| location.clone())
                    })
            })
        })
        .collect();

    if !failures.is_empty() {
        return Err(failures.into_iter().collect());
    }

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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_and_remove;
    use crate::compiler::{
        Field,
        analysis::{flow_analysis::FlowAnalysis, witness_taint_inference::WitnessTaintInference},
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
        validate_and_remove(ssa, &witness_inference)
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
