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
/// A witnessed integer **is** a field element, and a `Cast` carries it through unchanged, as
/// `hlssa_to_r1cs::Value::cast` passes a linear combination through as it is. Narrowing one
/// therefore discards nothing, while the result is declared at a width that excludes bits still in
/// that field element. This means that the honest value fails a range-check that it should have
/// passed, and results in a circuit that no witness satisfies, so we use a bit window to truncate
/// it ahead of time.
///
/// The window is emitted **in front of** the cast rather than in place of it, because a `BitRange`
/// takes its result's type from its _source_: a window of the low 32 bits of an `int200` is an
/// `int200` carrying a value below `2^32`. Replacing the cast with one would retype the cast's
/// result and break every consumer.
///
/// It reads a value range, but run inside `witness_integer_ops` it would also meet the pairs that
/// phase mints for itself whose results postdate the analysis snapshot and so carry no range at
/// all. Every one of those would be wrapped in a second, redundant window. A phase of its own gets
/// a fresh analysis.
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
        // and the window would be constraints for nothing. This is the shape every program
        // compiled from Noir takes: the frontend emits `bit_range(v, 0, n)` and then the cast, and
        // the window's own transfer bounds the result by `2^n`.
        //
        // `proves_` rather than the bare predicate: a bottom range means "no execution reaches
        // here", and eliding a truncation on the strength of that is circular: the only reason the
        // value cannot occur would be the very check being skipped.
        //
        // Above `HOST_WORD_BITS` the range domain deliberately stops tracking, so a window there
        // is never discharged and a program that already narrowed by hand pays for a second one.
        // Sound but not minimal, and invisible to any width Noir can name.
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
/// Only the witness domain is rewritten. A pure value is computed rather than constrained, and
/// both backends truncate a narrowing cast at every width.
///
/// Asked of the snapshot with `try_`, which is what makes this rule **idempotent**: a value it
/// minted itself has no entry there, so meeting its own output declines rather than wrapping it in
/// a second window. The driver does not re-offer an emitted instruction today, and this is what
/// keeps that from being load-bearing.
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
