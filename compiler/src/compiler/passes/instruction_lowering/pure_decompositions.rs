use super::{InstructionLoweringRule, LoweringContext};
use crate::compiler::ssa::hlssa::{
    OpCode,
    builder::{HLBlockEmitter, HLEmitter},
};

pub(super) struct LowerPureDecompositions;

impl InstructionLoweringRule for LowerPureDecompositions {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (op, guard) = match instruction {
            OpCode::Guard { condition, inner } => (inner.as_ref(), Some(*condition)),
            op => (op, None),
        };
        let (value, max_bits) = match op {
            OpCode::ToBits { value, count, .. } => (*value, *count),
            // The radix lowerer separately asserts that a dynamic radix equals 256.
            OpCode::ToRadix { value, count, .. } => (*value, count.saturating_mul(8)),
            _ => return false,
        };
        if context.types().get_value_type(value).is_witness_of()
            || max_bits >= b.field().field_bit_size() as usize
        {
            return false;
        }
        // Witness decompositions already constrain recomposition. Pure ones need only a
        // runtime/constant check; inserting it after witness inference adds no circuit work.
        let check = OpCode::Rangecheck { value, max_bits };
        b.emit(match guard {
            Some(condition) => OpCode::Guard {
                condition,
                inner: Box::new(check),
            },
            None => check,
        });
        b.emit(instruction.clone());
        true
    }
}
