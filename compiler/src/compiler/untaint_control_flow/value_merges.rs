//! Coordinate function lowering and emit deferred branch-result merges.
//!
//! The CFG phase supplies values, result types, and branch predicates. This phase
//! owns strategy selection and emission for every value type. The function driver
//! directly sequences provenance capture, linearization, and merge emission.
//!
//! Its intermediate state remains local; there is no callback or capture/finalize API.

use crate::collections::HashSet;
use mavros_int_semantics::IntBits;

use super::{UntaintControlFlow, emit_value_conversion, get_witness_or_pure};
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo, Types},
        value_range_analysis::ValueRangeAnalysis,
        witness_info::WitnessType,
        witness_taint_inference::WitnessTaintInference,
    },
    ssa::{
        BlockId, SourceLocation, Terminator, ValueId,
        hlssa::{
            CastTarget, HLFunction, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
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
    /// Keep provenance and pending merges private while sharing module analyses.
    pub(super) fn lower_functions(
        &mut self,
        ssa: &mut HLSSA,
        witness: &WitnessTaintInference,
        flow: &FlowAnalysis,
        types: &TypeInfo,
    ) {
        let mut pending = Vec::new();
        let mut unknown = HashSet::default();
        for id in ssa.get_function_ids().collect::<Vec<_>>() {
            let Some(wt) = witness.try_get_function_witness_type(id) else { continue };
            let mut function = ssa.take_function(id);
            let cfg = flow.get_function_cfg(id);
            let merge_blocks = function
                .get_blocks()
                .filter_map(|(block, body)| {
                    let Some(Terminator::JmpIf(cond, lhs, rhs)) = body.get_terminator() else {
                        return None;
                    };
                    if get_witness_or_pure(wt, *cond) != WitnessType::Witness {
                        return None;
                    }
                    let merge = cfg.get_merge_point(*block);
                    (merge != *lhs && merge != *rhs).then_some(merge)
                })
                .collect();
            let mut sparse =
                SparseArrayMerge::has_candidates(&function, types.get_function(id), &merge_blocks)
                    .then(|| SparseArrayMerge::new(&function, types.get_function(id), ssa));
            let merges = self.linearize_function(
                id,
                &mut function,
                ssa,
                wt,
                flow,
                Some(types.get_function(id)),
            );
            if let Some(sparse) = sparse.as_mut() {
                sparse.capture_guards(&function);
            }
            // Placeholder arguments describe only one arm; do not infer ranges
            // from them, including through values that depend on these parameters.
            for merge in &merges {
                unknown.extend(
                    function
                        .get_block(merge.destination)
                        .get_parameters()
                        .map(|(id, _)| *id),
                );
            }
            ssa.put_function(id, function);
            pending.push((id, sparse, merges));
        }
        // Analyze the rewritten CFG once, after all function signatures and calls
        // agree. Pre-linearization branch facts are unsafe for hoisted accesses.
        let mut ranges = pending
            .iter()
            .any(|(_, sparse, _)| sparse.is_some())
            .then(|| {
                let flow = FlowAnalysis::run(ssa);
                let types = Types::new().run(ssa, &flow);
                ValueRangeAnalysis::new().run_with_unknown_parameters(ssa, &flow, &types, &unknown)
            });
        for (id, mut sparse, merges) in pending {
            if let Some(sparse) = sparse.as_mut() {
                sparse.ranges = Some(ranges.as_mut().unwrap().take_function(id));
            }
            let mut function = ssa.take_function(id);
            emit_merges(
                &mut function,
                ssa,
                Some(types.get_function(id)),
                sparse.as_mut(),
                merges,
            );
            if let Some(sparse) = sparse {
                sparse.remove_redundant_updates(&mut function, ssa);
            }
            ssa.put_function(id, function);
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
    result_type: &Type,
    lhs_type: &Type,
    rhs_type: &Type,
) -> ValueId {
    match &result_type.expr {
        TypeExpr::Array(result_elem_type, size) => {
            // Match the identical-arm path's conversion contract, including lengths.
            CastTarget::assert_conversion(lhs_type, result_type);
            CastTarget::assert_conversion(rhs_type, result_type);
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
                let idx = builder.int_const(IntBits::from_u128(32, i as u128));
                let lhs_elem = builder.array_get(lhs, idx);
                let rhs_elem = builder.array_get(rhs, idx);
                let selected = emit_merge_select(
                    builder,
                    cond,
                    lhs_elem,
                    rhs_elem,
                    result_elem_type,
                    lhs_elem_type,
                    rhs_elem_type,
                );
                elems.push(selected);
            }
            let result = builder.fresh_value();
            builder.emit(OpCode::MkSeq {
                result,
                elems,
                seq_type: SequenceTargetType::Array(*size),
                elem_type: *result_elem_type.clone(),
            });
            result
        }
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        TypeExpr::WitnessOf(_) | TypeExpr::Field | TypeExpr::Int(_) | TypeExpr::Slice(_) => {
            let lhs = emit_value_conversion(lhs, lhs_type, result_type, builder);
            let rhs = emit_value_conversion(rhs, rhs_type, result_type, builder);
            builder.select(cond, lhs, rhs)
        }
        TypeExpr::Ref(_) => panic!("Witness select on Ref type not supported"),
        TypeExpr::Function(_) => panic!("Witness select on Function type not supported"),
        TypeExpr::Blob(..) => panic!("Witness select on Blob type not supported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::analysis::types::Types;

    #[test]
    fn sparse_array_merge_emits_nested_replay() {
        let mut ssa = HLSSA::with_main("nested_replay".into());
        let id = ssa.get_unique_entrypoint_id();
        let mut function = ssa.take_function(id);
        let entry = function.get_entry_id();
        let then_block = function.add_block();
        let else_block = function.add_block();
        let merge = function.add_block();
        let typ = Type::field().array_of(3).array_of(128);
        function.add_return_type(typ.clone());
        {
            let mut fb = HLFunctionBuilder::new(&mut function, &mut ssa);
            let mut b = fb.test_block(entry);
            let base = b.add_parameter(typ.clone());
            let index = b.add_parameter(Type::int(32));
            let enabled = b.add_parameter(Type::int(1));
            let value = b.add_parameter(Type::field());
            let condition = b.cast_to_witness_of(enabled);
            b.set_terminator(Terminator::JmpIf(condition, then_block, else_block));
            drop(b);
            let mut b = fb.test_block(then_block);
            let row = b.array_get(base, index);
            let zero = b.int_const(IntBits::from_u128(32, 0));
            let row = b.array_set(row, zero, value);
            let updated = b.array_set(base, index, row);
            b.set_terminator(Terminator::Jmp(merge, vec![updated]));
            drop(b);
            fb.test_block(else_block)
                .set_terminator(Terminator::Jmp(merge, vec![base]));
            let mut b = fb.test_block(merge);
            let result = b.add_parameter(typ);
            b.set_terminator(Terminator::Return(vec![result]));
        }
        ssa.put_function(id, function);
        let flow = FlowAnalysis::run(&ssa);
        let mut witness = WitnessTaintInference::new();
        witness.run(&mut ssa, &flow);
        let ssa = UntaintControlFlow::new().run(ssa, &witness);
        let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
        // Linearization guards the source writes. An unguarded inner-row write
        // therefore proves that merge emission actually chose nested replay.
        assert!(
            ssa.iter_functions().any(|(id, f)| {
                f.get_blocks().any(|(_, block)| {
                    block.get_instructions().any(|op| {
                        matches!(op, OpCode::ArraySet { array, .. }
                    if matches!(types.get_function(*id).get_value_type(*array).expr,
                        TypeExpr::Array(_, 3)))
                    })
                })
            }),
            "the small SSA program must exercise inner-array replay"
        );
    }

    #[test]
    fn identical_and_distinct_arms_share_the_conversion_contract() {
        use crate::compiler::ssa::hlssa::builder::HLInstrBuilder;
        use std::panic::{AssertUnwindSafe, catch_unwind};
        for (source, target, accepted) in [
            (Type::int(32), Type::int(32), true),
            (Type::int(32), Type::witness_of(Type::int(32)), true),
            (
                Type::int(32).array_of(2),
                Type::witness_of(Type::int(32)).array_of(2),
                true,
            ),
            (
                Type::int(32).slice_of(),
                Type::witness_of(Type::int(32)).slice_of(),
                true,
            ),
            (Type::int(8), Type::int(32), false),
            (Type::int(32), Type::field(), false),
            (Type::witness_of(Type::int(32)), Type::int(32), false),
            (Type::field().array_of(2), Type::field().array_of(3), false),
        ] {
            assert_eq!(
                catch_unwind(|| CastTarget::assert_conversion(&source, &target)).is_ok(),
                accepted
            );
            for identical in [false, true] {
                let mut ssa = HLSSA::with_main("conversion".into());
                let mut function = ssa.take_function(ssa.get_unique_entrypoint_id());
                let lhs = ssa.fresh_value();
                let rhs = ssa.fresh_value();
                let condition = ssa.fresh_value();
                let mut instructions = Vec::new();
                let mut b = HLInstrBuilder::new(
                    &mut function,
                    &mut ssa,
                    &mut instructions,
                    SourceLocation::synthetic("conversion"),
                );
                let result = catch_unwind(AssertUnwindSafe(|| {
                    if identical {
                        emit_value_conversion(lhs, &source, &target, &mut b)
                    } else {
                        emit_merge_select(&mut b, condition, lhs, rhs, &target, &source, &source)
                    }
                }));
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "{source} -> {target}, identical={identical}"
                );
            }
        }
    }

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
