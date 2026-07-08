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
            CastTarget, HLSSA, OpCode, SequenceTargetType, SliceOpDir, Type, TypeExpr,
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

                    // Every instruction scopes its own location below; emitting outside a scope
                    // is an ICE.
                    let mut emitter = fb
                        .block(bid)
                        .with_scoped_source_locations("lower_map_casts");
                    for instruction in instructions {
                        let location = instruction.location().clone();
                        emitter.emit_with_location(location, |emitter| {
                            match instruction.payload() {
                                OpCode::Cast {
                                    result,
                                    value,
                                    target: CastTarget::Map(inner),
                                } => {
                                    let src_type = type_info.get_value_type(value).clone();
                                    let converted = lower_map(emitter, value, &src_type, &inner);
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
                        });
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
            let elem_tgt = inner.result_type(elem_src);
            let initial = e.default_value(&elem_tgt.clone().array_of(*len));
            let results = e.build_counted_loop(
                *len,
                vec![(initial, elem_tgt.array_of(*len))],
                |e, i, accs| {
                    let elem = e.array_get(value, i);
                    let converted = apply_elem_cast(e, elem, elem_src, inner);
                    vec![e.array_set(accs[0], i, converted)]
                },
            );
            results[0]
        }
        TypeExpr::Slice(elem_src) => {
            let elem_tgt = inner.result_type(elem_src);
            let len = e.slice_len(value);
            let empty = e.mk_seq(Vec::new(), SequenceTargetType::Slice, elem_tgt.clone());
            let const_0 = e.u_const(32, 0);
            let const_1 = e.u_const(32, 1);
            let results = e.build_loop(
                vec![(const_0, Type::u(32)), (empty, elem_tgt.slice_of())],
                |b, params| b.lt(params[0], len),
                |e, params| {
                    let elem = e.array_get(value, params[0]);
                    let converted = apply_elem_cast(e, elem, elem_src, inner);
                    let pushed = e.slice_push(params[1], vec![converted], SliceOpDir::Back);
                    let next_i = e.add(params[0], const_1);
                    vec![next_i, pushed]
                },
            );
            results[1]
        }
        other => panic!("Map cast over non-sequence type {:?}", other),
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
