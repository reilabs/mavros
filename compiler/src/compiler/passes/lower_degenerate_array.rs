//! Makes a dynamic index into a degenerate array constant.
//!
//! Nothing below HLSSA can represent a dynamically selected *handle* so such an array has to be gone before untaint runs.
//!
//! At two lengths the index carries no information, so it can simply be replaced.
//!
//! - **One cell.** An in-bounds index can only be `0`, so the access becomes `d[0]`.
//! - **No cells.** Every index is out of bounds. The read is given a stand-in value of the
//!   element type instead, with a freshly allocated cell for every reference in it.

use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            CastTarget, CmpKind, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter, HLSSABuilder},
        },
    },
};

use mavros_int_semantics::IntBits;

pub struct LowerDegenerateArray {}

impl Pass for LowerDegenerateArray {
    fn name(&self) -> &'static str {
        "lower_degenerate_array"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>(), store.get::<FlowAnalysis>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl LowerDegenerateArray {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, types: &TypeInfo, flow: &FlowAnalysis) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for fid in fids {
            if !types.has_function(fid) {
                continue;
            }
            let fti = types.get_function(fid);
            let reachable: Vec<BlockId> = flow
                .get_function_cfg(fid)
                .get_domination_pre_order()
                .collect();
            sb.modify_function(fid, |fb| {
                for bid in reachable {
                    let terminator = fb.function.get_block_mut(bid).take_terminator();
                    let instructions = fb.function.get_block_mut(bid).take_instructions();
                    let mut emitter = fb
                        .block(bid)
                        .with_scoped_source_locations("lower_degenerate_array");
                    for instruction in instructions {
                        let (op, location) = instruction.take();
                        emitter.emit_with_location(location, |e| {
                            if !lower_access(e, fti, &op) {
                                e.emit(op);
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

struct DegenerateRefRead {
    result: ValueId,
    array: ValueId,
    index: ValueId,
    cells: usize,
    elem: Type,
}

fn get_degenerate_ref_read(
    ssa: &HLSSA,
    fti: &FunctionTypeInfo,
    op: &OpCode,
) -> Option<DegenerateRefRead> {
    let OpCode::ArrayGet {
        result,
        array,
        index,
    } = op
    else {
        return None;
    };
    if ssa.is_const(*index) {
        return None;
    }
    let TypeExpr::Array(elem, cells) = &fti.get_value_type(*array).strip_witness().expr else {
        return None;
    };
    if *cells > 1 {
        return None;
    }
    Some(DegenerateRefRead {
        result: *result,
        array: *array,
        index: *index,
        cells: *cells,
        elem: (**elem).clone(),
    })
}

fn stand_in(b: &mut HLBlockEmitter<'_>, ty: &Type) -> ValueId {
    if !ty.contains_ptrs() {
        return b.default_value(ty);
    }
    match &ty.expr {
        TypeExpr::Ref(pointee) => {
            let value = stand_in(b, pointee);
            b.alloc(value)
        }
        TypeExpr::Array(elem, len) => {
            let elems = (0..*len).map(|_| stand_in(b, elem)).collect();
            b.mk_seq(elems, SequenceTargetType::Array(*len), (**elem).clone())
        }
        TypeExpr::Tuple(elem_types) => {
            let elems = elem_types.iter().map(|elem| stand_in(b, elem)).collect();
            b.mk_tuple(elems, elem_types.clone())
        }
        _ => ice!("no stand-in value for {ty}"),
    }
}

fn lower_access(b: &mut HLBlockEmitter<'_>, fti: &FunctionTypeInfo, op: &OpCode) -> bool {
    let Some(DegenerateRefRead {
        result,
        array,
        index,
        cells,
        elem,
    }) = get_degenerate_ref_read(b.ssa, fti, op)
    else {
        return false;
    };
    let TypeExpr::Int(bits) = fti.get_value_type(index).strip_witness().expr else {
        return false;
    };

    match cells {
        0 => {
            let lhs = b.field_const(0u64);
            let rhs = b.field_const(1u64);
            b.emit(OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            });
            let value = stand_in(b, &elem);
            b.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::Nop,
            });
        }
        _ => {
            let zero = b.int_const(IntBits::zero(bits));
            b.emit(OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: index,
                rhs: zero,
            });
            b.emit(OpCode::ArrayGet {
                result,
                array,
                index: zero,
            });
        }
    }
    true
}
