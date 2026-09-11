//! Coordinate function lowering and emit deferred branch-result merges.
//!
//! The CFG phase supplies values, result types, and branch predicates. This phase
//! owns strategy selection and emission for every value type. The function driver
//! directly sequences provenance capture, linearization, and merge emission.
//!
//! Its intermediate state remains local; there is no callback or capture/finalize API.

use super::{UntaintControlFlow, emit_value_conversion, get_witness_or_pure};
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::FunctionTypeInfo,
        witness_info::{FunctionWitnessType, WitnessType},
    },
    ssa::{
        BlockId, FunctionId, SourceLocation, Terminator, ValueId,
        hlssa::{
            HLFunction, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLEmitter, HLFunctionBuilder},
        },
    },
    util::ice_non_elided_tuple,
};

mod array_merge;
use array_merge::SparseArrayMerge;

/// A branch-result merge, independent of the strategy used to implement it.
pub(super) struct MergePoint {
    pub block: BlockId,
    pub destination: BlockId,
    pub condition: ValueId,
    pub not_condition: ValueId,
    pub then_active: ValueId,
    pub else_active: ValueId,
    pub location: SourceLocation,
    pub values: Vec<(ValueId, ValueId, Type)>,
}

impl UntaintControlFlow {
    /// Own the complete per-function lowering sequence, including its temporary snapshot.
    pub(super) fn lower_function(
        &mut self,
        function_id: FunctionId,
        function: &mut HLFunction,
        ssa: &mut HLSSA,
        function_wt: &FunctionWitnessType,
        flow_analysis: &FlowAnalysis,
        types: Option<&FunctionTypeInfo>,
    ) {
        let has_witness_branch = function.get_blocks().any(|(_, block)| {
            matches!(block.get_terminator(), Some(Terminator::JmpIf(cond, _, _))
                if get_witness_or_pure(function_wt, *cond) == WitnessType::Witness)
        });
        let mut sparse = types
            .filter(|_| has_witness_branch)
            .map(|types| SparseArrayMerge::new(function, types, ssa));
        let merges = self.linearize_function(
            function_id,
            function,
            ssa,
            function_wt,
            flow_analysis,
            types,
        );
        emit_merges(function, ssa, types, sparse.as_mut(), merges);
        if let Some(sparse) = sparse {
            sparse.remove_redundant_updates(function, ssa);
        }
    }
}

fn emit_merges(
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    types: Option<&FunctionTypeInfo>,
    mut sparse: Option<&mut SparseArrayMerge<'_>>,
    merges: Vec<MergePoint>,
) {
    for merge in merges {
        let mut fb = HLFunctionBuilder::new(function, ssa);
        let mut builder = fb.block(merge.block).with_source_location(merge.location);
        let mut args = Vec::with_capacity(merge.values.len());
        for (lhs, rhs, typ) in merge.values {
            if lhs == rhs {
                let source_type = types.map(|ti| ti.get_value_type(lhs)).unwrap_or(&typ);
                args.push(emit_value_conversion(lhs, source_type, &typ, &mut builder));
                continue;
            }
            let selected = sparse
                .as_deref_mut()
                .and_then(|sparse| {
                    sparse.try_emit(
                        &mut builder,
                        merge.condition,
                        merge.not_condition,
                        merge.then_active,
                        merge.else_active,
                        lhs,
                        rhs,
                        &typ,
                    )
                })
                .unwrap_or_else(|| {
                    let lhs_type = types.map(|ti| ti.get_value_type(lhs)).unwrap_or(&typ);
                    let rhs_type = types.map(|ti| ti.get_value_type(rhs)).unwrap_or(&typ);
                    emit_merge_select(
                        &mut builder,
                        merge.condition,
                        lhs,
                        rhs,
                        None,
                        &typ,
                        lhs_type,
                        rhs_type,
                    )
                });
            args.push(selected);
        }
        builder.set_terminator(Terminator::Jmp(merge.destination, args));
    }
}

/// Emit selects for merge point values, handling type conversion between
/// branch values and the expected merge param type. For arrays, does unrolled
/// element-wise select + cast. For scalars, emits Select with optional cast.
fn emit_merge_select(
    builder: &mut impl HLEmitter,
    cond: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    result: Option<ValueId>,
    result_type: &Type,
    lhs_type: &Type,
    rhs_type: &Type,
) -> ValueId {
    match &result_type.expr {
        TypeExpr::Array(result_elem_type, size) => {
            let lhs_elem_type = match &lhs_type.expr {
                TypeExpr::Array(e, _) => e.as_ref(),
                _ => panic!(
                    "emit_merge_select: expected array for lhs, got {:?}",
                    lhs_type
                ),
            };
            let rhs_elem_type = match &rhs_type.expr {
                TypeExpr::Array(e, _) => e.as_ref(),
                _ => panic!(
                    "emit_merge_select: expected array for rhs, got {:?}",
                    rhs_type
                ),
            };
            let mut elems = Vec::with_capacity(*size);
            for i in 0..*size {
                let idx = builder.int_const(32, i as u128);
                let lhs_elem = builder.array_get(lhs, idx);
                let rhs_elem = builder.array_get(rhs, idx);
                let selected = emit_merge_select(
                    builder,
                    cond,
                    lhs_elem,
                    rhs_elem,
                    None,
                    result_elem_type,
                    lhs_elem_type,
                    rhs_elem_type,
                );
                elems.push(selected);
            }
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.emit(OpCode::MkSeq {
                result,
                elems,
                seq_type: SequenceTargetType::Array(*size),
                elem_type: *result_elem_type.clone(),
            });
            result
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
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.emit(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Field | TypeExpr::Int(_) => {
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.emit(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Ref(_) => panic!("Witness select on Ref type not supported"),
        TypeExpr::Slice(_) => {
            let lhs = emit_value_conversion(lhs, lhs_type, result_type, builder);
            let rhs = emit_value_conversion(rhs, rhs_type, result_type, builder);
            let result = result.unwrap_or_else(|| builder.fresh_value());
            builder.emit(OpCode::Select {
                result,
                cond,
                if_t: lhs,
                if_f: rhs,
            });
            result
        }
        TypeExpr::Function => panic!("Witness select on Function type not supported"),
        TypeExpr::Blob(..) => panic!("Witness select on Blob type not supported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::analysis::types::Types;

    #[test]
    fn identical_arms_reuse_the_value_and_keep_required_conversion() {
        for pure in [false, true] {
            let mut ssa = HLSSA::with_main("merge".into());
            let fid = ssa.get_unique_entrypoint_id();
            let mut function = ssa.take_function(fid);
            let entry = function.get_entry_id();
            let destination = function.add_block();
            let target = Type::witness_of(Type::field()).array_of(128);
            let source = if pure {
                Type::field().array_of(128)
            } else {
                target.clone()
            };
            let value = ssa.fresh_value();
            let condition = ssa.fresh_value();
            let result = ssa.fresh_value();
            function.get_block_mut(entry).push_parameter(value, source);
            function
                .get_block_mut(entry)
                .push_parameter(condition, Type::witness_of(Type::int(1)));
            function
                .get_block_mut(entry)
                .set_terminator(Terminator::Jmp(destination, vec![value]));
            function
                .get_block_mut(destination)
                .push_parameter(result, target.clone());
            function
                .get_block_mut(destination)
                .set_terminator(Terminator::Return(vec![result]));
            function.add_return_type(target.clone());
            ssa.put_function(fid, function);
            let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
            let mut function = ssa.take_function(fid);
            emit_merges(
                &mut function,
                &mut ssa,
                Some(types.get_function(fid)),
                None,
                vec![MergePoint {
                    block: entry,
                    destination,
                    condition,
                    not_condition: condition,
                    then_active: condition,
                    else_active: condition,
                    location: SourceLocation::synthetic("identical_merge"),
                    values: vec![(value, value, target.clone())],
                }],
            );
            let instructions: Vec<_> = function.get_block(entry).get_instructions().collect();
            assert_eq!(instructions.len(), usize::from(pure));
            assert!(
                instructions
                    .iter()
                    .all(|op| matches!(op, OpCode::Cast { .. }))
            );
            let Some(Terminator::Jmp(_, args)) = function.get_block(entry).get_terminator() else {
                panic!("missing merge jump")
            };
            let merged = args[0];
            if !pure {
                assert_eq!(merged, value);
            }
            ssa.put_function(fid, function);
            let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
            assert_eq!(types.get_function(fid).get_value_type(merged), &target);
        }
    }
}
