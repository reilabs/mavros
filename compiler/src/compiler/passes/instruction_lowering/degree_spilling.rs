use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, OpCode, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerDegreeSpillingOps {}

impl InstructionLoweringRule for LowerDegreeSpillingOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        match instruction {
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result,
                lhs,
                rhs,
            } => self.lower_mul(b, context.types(), *result, *lhs, *rhs),
            _ => false,
        }
    }
}

impl LowerDegreeSpillingOps {
    pub fn new() -> Self {
        Self {}
    }

    fn lower_mul(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let lhs_type = function_type_info.get_value_type(lhs);
        let lhs_witness = lhs_type.is_witness_of();
        let rhs_witness = function_type_info.get_value_type(rhs).is_witness_of();

        if !lhs_witness && !rhs_witness {
            return false;
        }

        match lhs_type.strip_witness().expr {
            TypeExpr::U(_) | TypeExpr::I(_) => {
                panic!(
                    "witness integer multiplication should have been lowered by instruction_lowering"
                )
            }
            _ if lhs_witness && rhs_witness => {
                let lhs_plain = b.value_of(lhs);
                let rhs_plain = b.value_of(rhs);
                let mul_hint = b.mul(lhs_plain, rhs_plain);
                b.emit(OpCode::WriteWitness {
                    result: Some(result),
                    value: mul_hint,
                    pinned: false,
                });
                b.constrain(lhs, rhs, result);
                true
            }
            _ => false,
        }
    }
}
