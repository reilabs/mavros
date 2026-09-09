use crate::compiler::{
    ssa::{
        ValueId,
        hlssa::{
            OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
    util::ice_non_elided_tuple,
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessMemoryOps {}

impl InstructionLoweringRule for LowerWitnessMemoryOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::Guard { condition, inner } = instruction else {
            return false;
        };

        let OpCode::Store { ptr, value } = inner.as_ref() else {
            return false;
        };

        let value_type = context.types().get_value_type(*value).clone();
        let old_value = b.load(*ptr);
        let new_value = emit_select(b, *condition, *value, old_value, &value_type);
        b.store(*ptr, new_value);
        true
    }
}

/// Slice selects must be emitted *bare*.
///
/// This pass is the second producer of witness `Select`s on slices (`untaint_control_flow`'s
/// `emit_merge_select` also), and unlike that it runs *after* `InstructionLowering::pure_guards`.
/// This means that `LowerSideEffectFreeGuards`, which is what normally strips a `Select`'s guard,
/// will never see what this emits. `LowerSliceSelect` asserts its input is unguarded, so a guarded
/// slice select emitted from here would trip that assert.
///
/// The `Slice` arm below is therefore deliberately a plain `b.select(..)`: the guarded store's
/// condition is already folded in as the select's `cond`, and nothing here re-wraps it.
fn emit_select(
    b: &mut HLBlockEmitter<'_>,
    cond: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    typ: &Type,
) -> ValueId {
    match &typ.expr {
        TypeExpr::Array(elem_type, size) => {
            let elem_type = (**elem_type).clone();
            b.build_array_loop(*size, elem_type.clone(), |b, idx| {
                let lhs_elem = b.array_get(lhs, idx);
                let rhs_elem = b.array_get(rhs, idx);
                emit_select(b, cond, lhs_elem, rhs_elem, &elem_type)
            })
        }
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        TypeExpr::Field | TypeExpr::Int(_) | TypeExpr::WitnessOf(_) => b.select(cond, lhs, rhs),
        TypeExpr::Ref(_) => panic!("Witness select on Ref type not supported"),
        TypeExpr::Slice(_) => b.select(cond, lhs, rhs),
        TypeExpr::Function => panic!("Witness select on Function type not supported"),
        TypeExpr::Blob(..) => panic!("Witness select on Blob type not supported"),
    }
}

impl LowerWitnessMemoryOps {
    pub fn new() -> Self {
        Self {}
    }
}
