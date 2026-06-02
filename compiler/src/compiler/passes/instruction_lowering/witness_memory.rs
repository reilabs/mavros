use crate::compiler::ssa::hlssa::{
    OpCode,
    builder::{HLBlockEmitter, HLEmitter},
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessMemoryOps {}

impl InstructionLoweringRule for LowerWitnessMemoryOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        _context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::Guard { condition, inner } = instruction else {
            return false;
        };

        let OpCode::Store { ptr, value } = inner.as_ref() else {
            return false;
        };

        let old_value = b.load(*ptr);
        let new_value = b.select(*condition, *value, old_value);
        b.store(*ptr, new_value);
        true
    }
}

impl LowerWitnessMemoryOps {
    pub fn new() -> Self {
        Self {}
    }
}
