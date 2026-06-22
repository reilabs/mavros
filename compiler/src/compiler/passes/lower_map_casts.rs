//! Lowers composite `Cast { target: Map(_) }` instructions into explicit
//! element-wise conversion loops.
//!
//! `Map` casts are emitted at typed-slot boundaries (by `UntaintControlFlow`
//! and `WitnessLowering`) so that the mid-pipeline optimization passes see a
//! single opaque cast instead of spilled loops. This pass runs late, right
//! before codegen needs the conversions to be explicit:
//!
//! - AD pipelines: after `WitnessLowering`, where `WitnessOf` values become
//!   witness references and the conversion is a real representation change.
//! - The witgen pipeline never runs this pass: `StripWitnessOf` erases
//!   witness-representation-only maps entirely (no loops at runtime).
//! - R1CS generation executes `Map` casts symbolically as pass-through clones.

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            CastTarget, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter, HLSSABuilder},
        },
    },
};

pub struct LowerMapCasts {}

impl Pass for LowerMapCasts {
    fn name(&self) -> &'static str {
        "lower_map_casts"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>(), store.get::<FlowAnalysis>());
    }
}

impl LowerMapCasts {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo, flow_analysis: &FlowAnalysis) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let type_info = type_info.get_function(function_id);
            let cfg = flow_analysis.get_function_cfg(function_id);
            let block_ids: Vec<BlockId> = cfg.get_domination_pre_order().collect();
            sb.modify_function(function_id, |fb| {
                for bid in block_ids {
                    let terminator = fb.function.get_block_mut(bid).take_terminator();
                    let instructions = fb.function.get_block_mut(bid).take_instructions();

                    let mut emitter = fb.block(bid);
                    for instruction in instructions {
                        match instruction {
                            OpCode::Cast {
                                result,
                                value,
                                target: CastTarget::Map(inner),
                            } => {
                                let src_type = type_info.get_value_type(value).clone();
                                let converted = lower_map(&mut emitter, value, &src_type, &inner);
                                // Bind the original result id to the loop's
                                // output; the Nop cast is a pure alias.
                                emitter.emit(OpCode::Cast {
                                    result,
                                    value: converted,
                                    target: CastTarget::Nop,
                                });
                            }
                            OpCode::Guard { condition, inner } => {
                                if matches!(
                                    inner.as_ref(),
                                    OpCode::Cast {
                                        target: CastTarget::Map(_),
                                        ..
                                    }
                                ) {
                                    panic!(
                                        "ICE: guarded Map cast reached LowerMapCasts; \
                                         LowerSideEffectFreeGuards should have unwrapped it"
                                    );
                                }
                                emitter.emit(OpCode::Guard { condition, inner });
                            }
                            other => emitter.emit(other),
                        }
                    }
                    if let Some(terminator) = terminator {
                        emitter.set_terminator(terminator);
                    }
                }
            });
        }
    }
}

/// Lower one `Map` cast over `value` (of sequence type `src_type`) into a loop
/// that applies `inner` to every element. Returns the converted sequence.
fn lower_map(
    e: &mut HLBlockEmitter<'_>,
    value: ValueId,
    src_type: &Type,
    inner: &CastTarget,
) -> ValueId {
    match &src_type.expr {
        TypeExpr::Array(elem_src, len) => {
            // Convert through a slice rather than emitting a counted loop over
            // the fixed-size array directly. A counted loop's `array_get`/
            // `array_set` lower to per-size lookup helper functions in codegen,
            // so one is generated for every distinct array length. The slice
            // loop below uses size-independent dynamic indexing, so a single
            // copy of the conversion code is shared across all array sizes.
            // Both alias casts are free at runtime (arrays and slices share the
            // same heap layout).
            let slice_src = elem_src.as_ref().clone().slice_of();
            let as_slice = e.cast_to(CastTarget::ArrayToSlice, value);
            let converted = lower_map(e, as_slice, &slice_src, inner);
            e.cast_to(CastTarget::SliceToArray(*len), converted)
        }
        TypeExpr::Slice(elem_src) => {
            // Pre-size the result slice to the source length and fill it with
            // `ArraySet` (rather than growing it with `SlicePush`): the target
            // backends implement a dynamic-length allocation + indexed stores,
            // but not slice growth.
            let elem_tgt = inner.result_type(elem_src);
            let len = e.slice_len(value);
            let fill = default_fill(e, &elem_tgt);
            let initial = e.mk_repeated_dyn(fill, len, elem_tgt.clone());
            let const_0 = e.u_const(32, 0);
            let const_1 = e.u_const(32, 1);
            let results = e.build_loop(
                vec![(const_0, Type::u(32)), (initial, elem_tgt.slice_of())],
                |b, params| b.lt(params[0], len),
                |e, params| {
                    let elem = e.array_get(value, params[0]);
                    let converted = apply_elem_cast(e, elem, elem_src, inner);
                    let updated = e.array_set(params[1], params[0], converted);
                    let next_i = e.add(params[0], const_1);
                    vec![next_i, updated]
                },
            );
            results[1]
        }
        other => panic!("Map cast over non-sequence type {:?}", other),
    }
}

/// A valid, droppable default value of `ty` to pre-fill a freshly allocated
/// slice slot before it is overwritten. Mirrors `default_value` but also covers
/// slice element types (an empty slice), which arise under nested `Map` casts.
fn default_fill(e: &mut HLBlockEmitter<'_>, ty: &Type) -> ValueId {
    match &ty.expr {
        TypeExpr::Slice(inner) => {
            e.mk_seq(Vec::new(), SequenceTargetType::Slice, inner.as_ref().clone())
        }
        _ => e.default_value(ty),
    }
}

/// Apply a cast target to a single element inside a lowered map loop.
fn apply_elem_cast(
    e: &mut HLBlockEmitter<'_>,
    value: ValueId,
    src_type: &Type,
    target: &CastTarget,
) -> ValueId {
    match target {
        CastTarget::Map(inner) => lower_map(e, value, src_type, inner),
        other => e.cast_to(other.clone(), value),
    }
}
