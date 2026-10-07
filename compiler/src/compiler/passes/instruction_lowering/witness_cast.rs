//! Lowers a narrowing `Cast` of a witnessed integer into the bit window that truncates it.

use crate::compiler::ssa::{
    ValueId,
    hlssa::{
        CastTarget, OpCode, TypeExpr,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

use super::{InstructionLoweringRule, LoweringContext};

/// Puts the bit window that truncates in front of a narrowing `Cast` of a witnessed operand.
///
/// A witnessed integer **is** a field element, and a `Cast` passes it through unchanged (as
/// `hlssa_to_r1cs::Value::cast` passes a linear combination through). So a narrowing cast discards
/// nothing, yet its result is declared at a width that excludes bits still in the element. The
/// honest value would then fail a range check it should pass, leaving a circuit no witness
/// satisfies, so a bit window truncates it first.
///
/// The window goes **in front of** the cast rather than replacing it, because a `BitRange` takes
/// its result type from its _source_: the low 32 bits of an `int200` are an `int200` below `2^32`.
/// Replacing the cast would retype its result and break every consumer.
///
/// This rule reads value ranges. Inside `witness_integer_ops` it would also meet the pairs that
/// phase creates, whose results postdate the analysis snapshot and so have no range; each would
/// get a second, redundant window. A separate phase gets a fresh analysis.
pub struct LowerWitnessNarrowingCast {}

impl LowerWitnessNarrowingCast {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for LowerWitnessNarrowingCast {
    fn default() -> Self {
        Self::new()
    }
}

impl InstructionLoweringRule for LowerWitnessNarrowingCast {
    fn needs_value_ranges(&self) -> bool {
        true
    }

    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(to_bits),
        } = instruction
        else {
            return false;
        };

        let Some(from_bits) = witnessed_int_width(context, *value) else {
            return false;
        };
        if from_bits <= *to_bits {
            return false;
        }

        // A value already inside the target width has nothing to discard, so the cast is a relabel
        // and the window would be constraints for nothing. A preceding bit window, for example,
        // bounds its result by `2^n`. Wide integer casts have already been lowered into limb
        // casts before this rule runs.
        //
        // `proves_` rather than the bare predicate: a bottom range means "no execution reaches
        // here", and eliding a truncation on the strength of that is circular: the only reason the
        // value cannot occur would be the very check being skipped.

        if context
            .urange(*value)
            .proves_fits_in_unsigned_bits(*to_bits)
        {
            return false;
        }

        let window = b.bit_range(*value, 0, *to_bits);
        b.emit(OpCode::Cast {
            result: *result,
            value: window,
            target: CastTarget::Int(*to_bits),
        });
        true
    }
}

/// The declared width of `value` where it is a witnessed integer, and [`None`] otherwise.
///
/// Only the witness domain is rewritten: a pure value is computed rather than constrained, and
/// both backends truncate a narrowing cast at every width.
///
/// Looked up in the snapshot with `try_`, which makes this rule **idempotent**: a value this rule
/// created has no entry there, so the rule declines it rather than wrapping it in a second window.
/// The driver does not re-offer emitted instructions, but this keeps the rule correct if it did.
fn witnessed_int_width(context: &LoweringContext<'_>, value: ValueId) -> Option<usize> {
    let ty = context.types().try_get_value_type(value)?;
    if !ty.is_witness_of() {
        return None;
    }
    match ty.strip_witness().expr {
        TypeExpr::Int(bits) => Some(bits),
        _ => None,
    }
}
