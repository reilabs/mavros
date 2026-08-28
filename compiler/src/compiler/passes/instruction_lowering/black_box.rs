//! Removes Noir `black_box` optimization barriers after high-level optimization has completed.

use crate::compiler::ssa::hlssa::{
    CastTarget, OpCode,
    builder::{HLBlockEmitter, HLEmitter},
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerBlackBox;

impl InstructionLoweringRule for LowerBlackBox {
    fn needs_value_ranges(&self) -> bool {
        false
    }

    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        _context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::BlackBox { result, value } = instruction else {
            return false;
        };

        b.emit(OpCode::Cast {
            result: *result,
            value: *value,
            target: CastTarget::Nop,
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler::{
        pass_manager::{AnalysisStore, Pass},
        passes::instruction_lowering::InstructionLowering,
        ssa::{
            Terminator,
            hlssa::{CastTarget, Constant, HLSSA, OpCode},
        },
    };

    #[test]
    fn lowers_barrier_to_identity_only_when_requested() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let value = ssa.add_const(Constant::Int(32, 9));
        let result = ssa.fresh_value();
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_test_instruction(OpCode::BlackBox { result, value });
        entry.set_terminator(Terminator::Return(vec![result]));

        InstructionLowering::black_box().run(&mut ssa, &AnalysisStore::new());

        let instructions: Vec<_> = ssa
            .get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .collect();
        assert!(matches!(
            instructions.as_slice(),
            [OpCode::Cast {
                result: lowered_result,
                value: lowered_value,
                target: CastTarget::Nop,
            }] if *lowered_result == result && *lowered_value == value
        ));
    }
}
