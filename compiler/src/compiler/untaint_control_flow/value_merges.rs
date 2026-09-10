//! Coordinate function lowering and emit deferred branch-result merges.
//!
//! The CFG phase supplies values, result types, and branch predicates. This phase
//! owns strategy selection and emission for every value type. The function driver
//! directly sequences provenance capture, linearization, and merge emission.
//! Its intermediate state remains local; there is no callback or capture/finalize API.

use super::{UntaintControlFlow, emit_value_conversion};
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis, types::FunctionTypeInfo, witness_info::FunctionWitnessType,
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
        let sparse = types.map(|types| SparseArrayMerge::new(function, types));
        let merges = self.linearize_function(
            function_id,
            function,
            ssa,
            function_wt,
            flow_analysis,
            types,
        );
        emit_merges(function, ssa, types, sparse.as_ref(), merges);
    }
}

fn emit_merges(
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    types: Option<&FunctionTypeInfo>,
    sparse: Option<&SparseArrayMerge<'_>>,
    merges: Vec<MergePoint>,
) {
    for merge in merges {
        let mut fb = HLFunctionBuilder::new(function, ssa);
        let mut builder = fb.block(merge.block).with_source_location(merge.location);
        let mut args = Vec::with_capacity(merge.values.len());
        for (lhs, rhs, typ) in merge.values {
            let selected = sparse
                .and_then(|sparse| {
                    sparse.try_emit(
                        &mut builder,
                        merge.condition,
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
