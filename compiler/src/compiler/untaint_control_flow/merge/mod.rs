//! Lower branch results to a single value with the required witness representation.
//!
//! Owns sparse-update matching, strategy selection, and the general merge fallback.
//! Construct before branch linearization so sparse matching sees original SSA
//! provenance. Callers provide branch predicates and retain ownership of CFG edges;
//! emission can advance the builder into newly created loop blocks.

use super::conversion::emit_value_conversion;
use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            HLFunction, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
    util::ice_non_elided_tuple,
};

mod sparse;
use sparse::SparseArrayMerge;

pub(super) struct MergeLowering<'a> {
    types: Option<&'a FunctionTypeInfo>,
    sparse: Option<SparseArrayMerge<'a>>,
}

impl<'a> MergeLowering<'a> {
    pub(super) fn new(function: &HLFunction, types: Option<&'a FunctionTypeInfo>) -> Self {
        Self {
            types,
            sparse: types.map(|types| SparseArrayMerge::new(function, types)),
        }
    }

    /// The arm predicates include enclosing guards; `condition` selects this merge's arm.
    pub(super) fn emit(
        &self,
        builder: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        then_active: ValueId,
        else_active: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        result_type: &Type,
    ) -> ValueId {
        if let Some(value) = self.sparse.as_ref().and_then(|sparse| {
            sparse.try_emit(
                builder,
                condition,
                then_active,
                else_active,
                lhs,
                rhs,
                result_type,
            )
        }) {
            return value;
        }
        let lhs_type = self
            .types
            .map(|types| types.get_value_type(lhs))
            .unwrap_or(result_type);
        let rhs_type = self
            .types
            .map(|types| types.get_value_type(rhs))
            .unwrap_or(result_type);
        emit_merge_select(
            builder,
            condition,
            lhs,
            rhs,
            result_type,
            lhs_type,
            rhs_type,
        )
    }
}

/// Emit selects for merge point values, handling type conversion between
/// branch values and the expected merge param type. Arrays use counted SSA loops
/// so the emitted code grows with nesting depth, rather than the number of leaves.
/// Each leaf is still selected independently; selecting an array pointer with a
/// witness condition would not constrain its contents. This bounds emitted code size;
/// execution and constraint generation still select every leaf of the array.
fn emit_merge_select(
    builder: &mut HLBlockEmitter<'_>,
    cond: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    result_type: &Type,
    lhs_type: &Type,
    rhs_type: &Type,
) -> ValueId {
    match &result_type.expr {
        TypeExpr::Array(result_elem_type, size) => {
            let rhs_elem_type = match &rhs_type.expr {
                TypeExpr::Array(e, _) => e.as_ref(),
                _ => panic!(
                    "emit_merge_select: expected array for rhs, got {:?}",
                    rhs_type
                ),
            };
            // Seed the accumulator from an arm rather than a fabricated default. This also
            // supports empty arrays and array elements (such as slices) without a default.
            let lhs = emit_value_conversion(lhs, lhs_type, result_type, builder);
            if *size == 0 {
                return lhs;
            }
            builder.build_counted_loop(
                *size,
                vec![(lhs, result_type.clone())],
                |builder, idx, accs| {
                    let lhs_elem = builder.array_get(lhs, idx);
                    let rhs_elem = builder.array_get(rhs, idx);
                    let selected = emit_merge_select(
                        builder,
                        cond,
                        lhs_elem,
                        rhs_elem,
                        result_elem_type,
                        result_elem_type,
                        rhs_elem_type,
                    );
                    vec![builder.array_set(accs[0], idx, selected)]
                },
            )[0]
        }
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        TypeExpr::WitnessOf(_) => {
            // Cast operands to WitnessOf if they aren't already
            let lhs = if !lhs_type.is_witness_of() {
                builder.cast_to_witness_of(lhs)
            } else {
                lhs
            };
            let rhs = if !rhs_type.is_witness_of() {
                builder.cast_to_witness_of(rhs)
            } else {
                rhs
            };
            builder.select(cond, lhs, rhs)
        }
        TypeExpr::Field | TypeExpr::Int(_) => builder.select(cond, lhs, rhs),
        TypeExpr::Ref(_) => panic!("Witness select on Ref type not supported"),
        TypeExpr::Slice(_) => {
            let lhs = emit_value_conversion(lhs, lhs_type, result_type, builder);
            let rhs = emit_value_conversion(rhs, rhs_type, result_type, builder);
            builder.select(cond, lhs, rhs)
        }
        TypeExpr::Function => panic!("Witness select on Function type not supported"),
        TypeExpr::Blob(..) => panic!("Witness select on Blob type not supported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        ssa::{
            Terminator,
            hlssa::{HLSSA, builder::HLFunctionBuilder},
        },
    };

    fn nested_merge_instruction_count(size: usize, mixed: bool) -> usize {
        let mut ssa = HLSSA::with_main("main".to_string());
        let fid = ssa.get_unique_entrypoint_id();
        let mut function = ssa.take_function(fid);
        let entry = function.get_entry_id();
        let result_type = Type::witness_of(Type::int(64)).array_of(3).array_of(size);
        let lhs_type = if mixed {
            Type::int(64).array_of(3).array_of(size)
        } else {
            result_type.clone()
        };
        let cond = ssa.fresh_value();
        let lhs = ssa.fresh_value();
        let rhs = ssa.fresh_value();
        function
            .get_block_mut(entry)
            .push_parameter(cond, Type::witness_of(Type::int(1)));
        function
            .get_block_mut(entry)
            .push_parameter(lhs, lhs_type.clone());
        function
            .get_block_mut(entry)
            .push_parameter(rhs, result_type.clone());
        function.add_return_type(result_type.clone());
        let mut fb = HLFunctionBuilder::new(&mut function, &mut ssa);
        let mut b = fb.test_block(entry);
        let result = emit_merge_select(
            &mut b,
            cond,
            lhs,
            rhs,
            &result_type,
            &lhs_type,
            &result_type,
        );
        b.set_terminator(Terminator::Return(vec![result]));
        drop(b);
        let count = function
            .get_blocks()
            .map(|(_, b)| b.get_instructions().count())
            .sum();
        ssa.put_function(fid, function);
        let flow = FlowAnalysis::run(&ssa);
        let types = Types::new().run(&ssa, &flow);
        assert_eq!(types.get_function(fid).get_value_type(result), &result_type);
        count
    }

    #[test]
    fn nested_merge_code_size_does_not_grow_with_array_length() {
        for mixed in [false, true] {
            let small = nested_merge_instruction_count(2, mixed);
            let large = nested_merge_instruction_count(1024, mixed);
            assert_eq!(small, large);
            assert!(large < 40, "nested merge emitted {large} instructions");
        }
    }

    #[test]
    fn empty_merge_emits_no_loop_or_element_access() {
        assert_eq!(nested_merge_instruction_count(0, false), 0);
        // A pure arm still needs a type conversion to the witness element type.
        assert_eq!(nested_merge_instruction_count(0, true), 1);
    }
}

#[cfg(test)]
mod differential;
