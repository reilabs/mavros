use crate::compiler::{
    ssa::{
        ValueId,
        hlssa::{
            OpCode, SequenceTargetType, Type, TypeExpr,
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

fn emit_select(
    b: &mut HLBlockEmitter<'_>,
    cond: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    typ: &Type,
) -> ValueId {
    match &typ.expr {
        TypeExpr::Array(elem_type, size) => {
            let mut elems = Vec::with_capacity(*size);
            for i in 0..*size {
                let idx = b.u_const(32, i as u128);
                let lhs_elem = b.array_get(lhs, idx);
                let rhs_elem = b.array_get(rhs, idx);
                elems.push(emit_select(b, cond, lhs_elem, rhs_elem, elem_type));
            }
            b.mk_seq(elems, SequenceTargetType::Array(*size), *elem_type.clone())
        }
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::WitnessOf(_) => {
            b.select(cond, lhs, rhs)
        }
        TypeExpr::Ref(_) => panic!("Witness select on Ref type not supported"),
        TypeExpr::Slice(_) => panic!("Witness select on Slice type not supported"),
        TypeExpr::Function => panic!("Witness select on Function type not supported"),
        TypeExpr::Blob(..) => panic!("Witness select on Blob type not supported"),
    }
}

impl LowerWitnessMemoryOps {
    pub fn new() -> Self {
        Self {}
    }
}
