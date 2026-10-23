//! Drop every range check no field element can fail, which no lowering decomposes: one wider than
//! the widest integer the field carries injectively is above the modulus. Noir writes one only
//! through `assert_max_bit_size`, whose `static_assert` rejects the program anyway.

use crate::compiler::{
    passes::{
        instruction_lowering::{InstructionLoweringRule, LoweringContext},
        shared::limbs::widest_injective_int_bits,
    },
    ssa::hlssa::{
        OpCode,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

pub(super) struct VacuousRangechecks;

impl InstructionLoweringRule for VacuousRangechecks {
    fn needs_value_ranges(&self) -> bool {
        false
    }

    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        _context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        matches!(instruction, OpCode::Rangecheck { max_bits, .. }
            if *max_bits > widest_injective_int_bits(b.field()))
    }
}
