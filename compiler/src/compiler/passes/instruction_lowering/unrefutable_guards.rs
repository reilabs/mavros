//! Removes guards around operations that are safe to execute unconditionally.

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::hlssa::{
        BinaryArithOpKind, OpCode, TypeExpr,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerUnrefutableGuards {}

impl InstructionLoweringRule for LowerUnrefutableGuards {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::Guard { inner, .. } = instruction else {
            return false;
        };

        if self.is_unrefutable(inner, context.types()) {
            b.emit(inner.as_ref().clone());
            true
        } else {
            false
        }
    }
}

impl LowerUnrefutableGuards {
    pub fn new() -> Self {
        Self {}
    }

    fn is_unrefutable(&self, op: &OpCode, type_info: &FunctionTypeInfo) -> bool {
        match op {
            OpCode::Cmp { .. } => true,
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add | BinaryArithOpKind::Sub | BinaryArithOpKind::Mul,
                lhs,
                ..
            } => {
                matches!(
                    type_info.get_value_type(*lhs).strip_witness().expr,
                    TypeExpr::Field
                )
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor,
                ..
            } => true,
            OpCode::BinaryArithOp {
                kind:
                    BinaryArithOpKind::Div
                    | BinaryArithOpKind::Mod
                    | BinaryArithOpKind::Shl
                    | BinaryArithOpKind::Shr,
                ..
            } => false,
            OpCode::Cast { .. }
            | OpCode::SExt { .. }
            | OpCode::BitRange { .. }
            | OpCode::Not { .. }
            | OpCode::MkSeq { .. }
            | OpCode::MkRepeated { .. }
            | OpCode::MkTuple { .. }
            | OpCode::TupleProj { .. }
            | OpCode::Alloc { .. }
            | OpCode::Load { .. }
            | OpCode::SlicePush { .. }
            | OpCode::SliceLen { .. }
            | OpCode::Select { .. }
            | OpCode::ToBits { .. }
            | OpCode::ToRadix { .. }
            | OpCode::ValueOf { .. }
            | OpCode::WriteWitness { .. }
            | OpCode::FreshWitness { .. }
            | OpCode::NextDCoeff { .. }
            | OpCode::BumpD { .. }
            | OpCode::MulConst { .. }
            | OpCode::ReadGlobal { .. }
            | OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::Spread { .. }
            | OpCode::Unspread { .. }
            | OpCode::Todo { .. } => true,
            OpCode::Store { .. }
            | OpCode::Assert { .. }
            | OpCode::AssertCmp { .. }
            | OpCode::AssertR1C { .. }
            | OpCode::Call { .. }
            | OpCode::ArrayGet { .. }
            | OpCode::ArraySet { .. }
            | OpCode::MemOp { .. }
            | OpCode::Constrain { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. }
            | OpCode::Rangecheck { .. } => false,
            OpCode::Guard { .. } => panic!("nested Guard not expected"),
        }
    }
}
