//! The failure condition of a shift, and what 'in range' means.
//!
//! A shift with a **witness** operand is not handled here: its lowering performs the same rejection
//! but cannot build it from a pure comparison.
//!
//! Past the single cell that lowering is `WideWitnessInts`' limb-wise shift. A witnessed amount is
//! bounded by its decomposition into whole limbs and a remainder, plus an explicit check where that
//! decomposition reaches past the width. A pure amount is compared as here, under the shift's guard
//! if it has one, and a known amount past the width is an assertion that cannot hold.
//!
//! Within the single cell, `LowerWitnessBitwiseOps::lower_shift` emits `emit_shift_amount_check`
//! above its two lowerings, except on the route that already carries the rejection:
//!
//! - A **witness amount** the range domain has not pinned, at a power-of-two width, gets its bound
//!   free from the powers-of-two lookup that reads its factor: the table's keys are exactly the
//!   legal amounts, so an out-of-range one has no row and no check is emitted. At any other width
//!   the table has more rows than legal amounts, so the explicit check is emitted too.
//! - A **pure amount** always pays the explicit check. An unsigned left-hand side takes
//!   `lower_constant_amount_shift` and a signed one `lower_general_shift` (the former is
//!   unsigned-only), and the check sits above that split because both need it. This is the common
//!   case, covered by `witness_shift_amount_oob_fails`.
//! - A **witness amount the range domain has pinned**, with an unsigned left-hand side, is treated
//!   as a constant: `shift_amount_pinned_to` reports the literal and the shift is lowered as if it
//!   had been written that way. The check is emitted and discharges itself against the same range.
//!
//! A pinned amount with a _signed_ left-hand side still takes the general lowering, since only the
//! constant-amount lowering can use the literal, so it falls under the first bullet: the table and
//! no check.
//!
//! Both witness lowerings compare a pure amount through [`emit_pure_shift_amount_check`], which
//! reads the raw pattern where the guard IR reads the shift's signedness. Witness amounts are
//! checked by constraints instead, so the guard IR and the witness lowerings are separate
//! implementations of one contract, kept in agreement by tests.

use mavros_int_semantics::IntBits;
use num_traits::ToPrimitive;

use crate::compiler::{
    analysis::value_range_analysis::ValueRange,
    ssa::{
        ValueId,
        hlssa::{CastTarget, CmpKind, OpCode, Type, TypeExpr, builder::HLEmitter},
    },
};

/// The width a shift is performed at, or `None` for an operand type that cannot be shifted.
///
/// The shift counterpart of [`super::divmod_guard::divmod_can_fail`], returning the width the
/// amount is checked against. Every width is admitted: every evaluator reduces the amount modulo
/// `bits`, and every witness lowering checks `amount < bits`.
pub fn shift_operand_bits(lhs_type: &Type) -> Option<usize> {
    match lhs_type.strip_witness().expr {
        TypeExpr::Int(bits) => Some(bits),
        _ => None,
    }
}

/// Whether the range domain **proves** this shift's amount is in range, so the check
/// `emit_valid_shift_cond` would build is dead and need not be emitted at all.
///
/// Asked of the raw pattern rather than of a chosen reading: an amount below the width is
/// non-negative under the signed reading too, so one answer serves both shift kinds. See
/// [`ValueRange::proves_shift_amount_below`].
///
/// Note that ⊥ answers `false`, like every other `proves_*` query. A shift whose amount range is ⊥
/// is exactly the shift the check is there to reject.
pub fn shift_amount_provably_in_range(amount: &ValueRange, bits: usize) -> bool {
    amount.proves_shift_amount_below(bits)
}

/// The one legal shift amount this range pins the value to, if it pins it to a single one at all.
///
/// A shift by a pinned amount can be lowered as though the amount had been _written_ as a literal,
/// whatever its type says. What a constant-amount lowering actually needs is that `1 << amount`
/// folds, and a minted literal gives it that whether or not the operand it stands in for was a
/// witness. `CastTarget::WitnessOf` is transparent to the range domain, so a witness-typed value
/// the domain has pinned is a literal in every sense but its type.
///
/// Only an amount that is _also_ in range is reported. An out-of-range one is a rejection, and is
/// left to whichever lowering already knows how to reject it; answering here would move that
/// rejection onto a different mechanism for no gain.
pub fn shift_amount_pinned_to(amount: &ValueRange, bits: usize) -> Option<u128> {
    if !shift_amount_provably_in_range(amount, bits) {
        return None;
    }
    let pinned = amount.proves_constant()?;
    Some(
        pinned
            .to_u128()
            .expect("a shift amount proved below the operand width fits in a u128"),
    )
}

/// The two facts [`emit_shift_amount_tests`] establishes about a shift amount.
struct ShiftAmountTests {
    /// `1` when the amount is below the operand width, under the shift's own reading.
    below_width: ValueId,

    /// `1` when the amount is negative, under the signed reading. Signed shifts only.
    negative: Option<ValueId>,
}

/// The two tests a shift amount needs to pass before either polarity can be built.
fn emit_shift_amount_tests(
    emitter: &mut impl HLEmitter,
    rhs: ValueId,
    bits: usize,
    signed: bool,
) -> ShiftAmountTests {
    // Both tests compare at `cmp_bits`, the operand width or 64 if that is wider, under the shift's
    // reading.
    let cmp_bits = bits.max(64);

    // The cast is the same under either reading (it widens by zero-extending raw bits) so it takes
    // no sign. `signed` still decides the two comparisons below.
    let rhs_cmp = emitter.cast_to(CastTarget::Int(cmp_bits), rhs);
    let rhs_bound = emitter.int_const(IntBits::from_u128(cmp_bits, bits as u128));
    let lt = CmpKind::lt(signed);
    let below_width = emitter.cmp(rhs_cmp, rhs_bound, lt);

    let negative = signed.then(|| {
        // Live wherever the amount is as wide as `cmp_bits`, and the only check that catches a
        // negative amount there. Noir gives a shift's amount the value's type
        // (`noir_tests/signed_shift` shifts an `i8` by an `i8`), so on an `i64` or wider the cast
        // above is an identity, not a widening. A negative amount keeps its sign bit, which
        // `below_width` does _not_ catch: `-1 s< 64` holds, so the bound test passes the shift.
        //
        // Below 64 bits it is dead, because the widening cast zero-extends the raw bits (`Cast`
        // masks; sign extension is the separate `SExt`) and so clears the sign bit at `cmp_bits`.
        // That holds only for the narrow widths, so the check stays.
        let zero = emitter.int_const(IntBits::zero(cmp_bits));
        emitter.cmp(rhs_cmp, zero, lt)
    });

    ShiftAmountTests {
        below_width,
        negative,
    }
}

/// The `u1` that is `1` exactly when the amount is **out of range**, for the guarded lowering to
/// branch on.
pub fn emit_invalid_shift_cond(
    emitter: &mut impl HLEmitter,
    rhs: ValueId,
    bits: usize,
    signed: bool,
) -> ValueId {
    let tests = emit_shift_amount_tests(emitter, rhs, bits, signed);
    let too_large = emitter.not(tests.below_width);
    match tests.negative {
        Some(negative) => emitter.or(negative, too_large),
        None => too_large,
    }
}

/// The `u1` that is `1` exactly when the amount is **in range**, the body of
/// [`emit_shift_amount_is_valid_assert`].
///
/// The complement of [`emit_invalid_shift_cond`], built from the same two tests rather than by
/// negating it: the unsigned case is then `below_width` itself, with no `Not` at all. This is
/// needed because `LowerWitnessBitwiseOps::lower_not` rewrites even a pure `Not` into a field
/// subtraction, so the negation that reads as free actually isn't.
fn emit_valid_shift_cond(
    emitter: &mut impl HLEmitter,
    rhs: ValueId,
    bits: usize,
    signed: bool,
) -> ValueId {
    let tests = emit_shift_amount_tests(emitter, rhs, bits, signed);
    match tests.negative {
        Some(negative) => {
            let non_negative = emitter.not(negative);
            emitter.and(non_negative, tests.below_width)
        }
        None => tests.below_width,
    }
}

/// Assert, under `guard` if there is one, that a **pure** amount's raw pattern is below `bits`.
///
/// This is the check a pure amount pays when a witness lowering, not `LowerPureGuards`, takes the
/// shift: `LowerWitnessBitwiseOps` within the single cell and `WideWitnessInts` past it. It reads
/// the pattern unsigned, which also rejects a negative amount, since the smallest negative
/// pattern, `2^(bits - 1)`, is at least `bits`.
pub fn emit_pure_shift_amount_check(
    emitter: &mut impl HLEmitter,
    guard: Option<ValueId>,
    rhs: ValueId,
    bits: usize,
) {
    let bound = emitter.int_const(IntBits::from_u128(bits, bits as u128));
    let below = emitter.cmp(rhs, bound, CmpKind::ULt);
    let one = emitter.int_const(IntBits::one(1));
    let check = OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: below,
        rhs: one,
    };
    emitter.emit(match guard {
        Some(condition) => OpCode::Guard {
            condition,
            inner: Box::new(check),
        },
        None => check,
    });
}

/// Emit `assert(valid(rhs))`: the unguarded form of the check.
///
/// The caller decides whether the shift itself follows: `LowerPureGuards` re-plants it, `DCE` does
/// not.
pub fn emit_shift_amount_is_valid_assert(
    emitter: &mut impl HLEmitter,
    rhs: ValueId,
    bits: usize,
    signed: bool,
) {
    let valid = emit_valid_shift_cond(emitter, rhs, bits, signed);
    let one_u1 = emitter.int_const(IntBits::one(1));
    emitter.emit(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: valid,
        rhs: one_u1,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::analysis::value_range_analysis::{Interval, Width};

    #[test]
    fn only_a_pinned_and_legal_shift_amount_is_reported() {
        let pinned = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(3, 3));
        assert_eq!(shift_amount_pinned_to(&pinned, 32), Some(3));

        // Pinned but not legal. The shift is a rejection, and reporting it here would swap the
        // mechanism that rejects it for a folded constant that quietly shifts by `amount & 31`.
        let too_large = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(32, 32));
        assert_eq!(shift_amount_pinned_to(&too_large, 32), None);

        // Negative is the same rejection under a different reading, and is caught by the same
        // in-range test rather than by inspecting the sign.
        let negative = ValueRange::from_signed(Width::Bits(8), Interval::closed(-1, -1));
        assert_eq!(shift_amount_pinned_to(&negative, 32), None);

        // Legal but not pinned: in range is not the same question as known.
        let small = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(0, 7));
        assert!(shift_amount_provably_in_range(&small, 32));
        assert_eq!(shift_amount_pinned_to(&small, 32), None);

        // ⊥ is neither, and answers `None` from the in-range test alone.
        let bottom = ValueRange::from_unsigned(Width::Bits(8), Interval::empty());
        assert_eq!(shift_amount_pinned_to(&bottom, 32), None);

        // A wide amount still fits the `u128` the literal is minted from, because it is only ever
        // reported below the operand width and nothing is wider than 128 bits.
        let wide = ValueRange::from_unsigned(Width::Bits(128), Interval::closed(127, 127));
        assert_eq!(shift_amount_pinned_to(&wide, 128), Some(127));
    }
}
