//! Removes guards around side-effect-free operations that are safe to execute unconditionally.

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::hlssa::{
        ArithGroup, OpCode, TypeExpr,
        builder::{HLBlockEmitter, HLEmitter},
    },
    util::{ice_non_elided_tuple, ice_unvalidated_assert_constant},
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerSideEffectFreeGuards {}

impl InstructionLoweringRule for LowerSideEffectFreeGuards {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::Guard { inner, .. } = instruction else {
            return false;
        };

        if self.can_drop_guard(inner, context.types()) {
            b.emit(inner.as_ref().clone());
            true
        } else {
            false
        }
    }
}

impl LowerSideEffectFreeGuards {
    pub fn new() -> Self {
        Self {}
    }

    fn can_drop_guard(&self, op: &OpCode, type_info: &FunctionTypeInfo) -> bool {
        match op {
            OpCode::Cmp { .. } => true,
            // Sign-blind by construction: what makes an `Add` droppable is that field addition
            // cannot fail, and what makes a `Div` undroppable is that a zero divisor can — neither
            // depends on how the operands are read.
            OpCode::BinaryArithOp { kind, lhs, .. } => match kind.group() {
                ArithGroup::Add | ArithGroup::Sub | ArithGroup::Mul => matches!(
                    type_info.get_value_type(*lhs).strip_witness().expr,
                    TypeExpr::Field
                ),
                ArithGroup::And | ArithGroup::Or | ArithGroup::Xor => true,
                ArithGroup::Div | ArithGroup::Rem | ArithGroup::Shl | ArithGroup::Shr => false,
            },
            OpCode::Cast { .. }
            | OpCode::SExt { .. }
            | OpCode::BitRange { .. }
            | OpCode::Not { .. }
            | OpCode::MkSeq { .. }
            | OpCode::MkSeqOfBlob { .. }
            | OpCode::MkRepeated { .. }
            | OpCode::Alloc { .. }
            | OpCode::Load { .. }
            | OpCode::SlicePush { .. }
            | OpCode::SliceLen { .. }
            | OpCode::Select { .. }
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
            OpCode::ToBits { value, .. } => !type_info.get_value_type(*value).is_witness_of(),
            OpCode::ToRadix { value, .. } => !type_info.get_value_type(*value).is_witness_of(),
            // Guard-elision judgment, not liveness. A pop from empty or an OOB insert/remove fails,
            // so running one unconditionally would fail a world whose guard is off.
            OpCode::SlicePop { .. } | OpCode::SliceInsert { .. } | OpCode::SliceRemove { .. } => {
                false
            }
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
            OpCode::AssertConstant { .. } => ice_unvalidated_assert_constant(),
            OpCode::MkTuple { .. } | OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } => {
                ice_non_elided_tuple()
            }
            OpCode::Guard { .. } => panic!("nested Guard not expected"),
        }
    }
}
