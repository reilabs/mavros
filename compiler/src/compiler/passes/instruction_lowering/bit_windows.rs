//! Rewrite a bit window of an integer too wide for one field element into a shift and two casts.
//!
//! The witnessed bit-window lowering rebuilds its source in one field element, so it takes only a
//! source the field carries injectively. This rule runs before untaint, where a window's
//! witness-ness is not known yet, so every wider window becomes `(value >> offset)`, narrowed to the
//! window's width and widened back, keeping its type. The shift is by a literal below the width, so
//! it cannot fail, and the representation lowers all three at any width.

use crate::compiler::{
    passes::{
        instruction_lowering::{InstructionLoweringRule, LoweringContext, integer_bits},
        shared::limbs::widest_injective_int_bits,
    },
    ssa::hlssa::{
        BinaryArithOpKind, CastTarget, OpCode,
        builder::{HLBlockEmitter, HLEmitter},
    },
};
use mavros_int_semantics::IntBits;

pub(super) struct WideBitWindows;

impl InstructionLoweringRule for WideBitWindows {
    fn needs_value_ranges(&self) -> bool {
        false
    }

    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::BitRange {
            result,
            value,
            offset,
            width,
        } = *instruction
        else {
            return false;
        };
        let Some(bits) = context
            .types()
            .try_get_value_type(value)
            .and_then(integer_bits)
        else {
            return false;
        };
        if bits <= widest_injective_int_bits(b.field()) {
            return false;
        }

        let shifted = if offset == 0 {
            value
        } else {
            let amount = b.int_const(IntBits::from_u128(bits, offset as u128));
            b.bin(BinaryArithOpKind::UShr, value, amount)
        };
        let window = b.cast_to(CastTarget::Int(width), shifted);
        b.emit(OpCode::Cast {
            result,
            value: window,
            target: CastTarget::Int(bits),
        });
        true
    }
}
