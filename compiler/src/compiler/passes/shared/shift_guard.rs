//! The failure condition of a shift and the decision on what 'in range' means.
//!
//! A shift with a **witness** operand is not handled here. `LowerWitnessBitwiseOps::lower_shift`
//! owes the same rejection but has to build it out of constraints rather than out of a pure
//! comparison, so it emits its own (`emit_shift_amount_check`, hoisted above its two lowerings).
//! That is a separate implementation of the same contract, so we rely on a pair of tests to ensure
//! they work the same: `noir_failure_tests/pure_shift_amount_oob_fails` and
//! `witness_shift_amount_oob_fails`.

use crate::compiler::{
    analysis::value_range_analysis::ValueRange,
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, CmpKind, OpCode, Type, TypeExpr, assert_signed_op_width, builder::HLEmitter,
        },
    },
};

/// The width a shift is performed at, or `None` for an operand type that cannot be shifted.
///
/// This is the shift's counterpart of [`super::divmod_guard::divmod_can_fail`], and it returns the
/// width rather than a `bool`: the amount is checked against the shifted value's width.
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
    if signed {
        // The check below compares at `cmp_bits`, which is 64 for every width this can reach; a
        // wider signed amount would need an `SLt` at that width, which nothing under HLSSA
        // currently provides.
        assert_signed_op_width(bits, "shift amount check");
    }
    let cmp_bits = bits.max(64);

    // The cast is the same under either reading (it widens by zero-extending raw bits) so it takes
    // no sign. `signed` still decides the two comparisons below.
    let rhs_cmp = emitter.cast_to(CastTarget::Int(cmp_bits), rhs);
    let rhs_bound = emitter.int_const(cmp_bits, bits as u128);
    let lt = CmpKind::lt(signed);
    let below_width = emitter.cmp(rhs_cmp, rhs_bound, lt);

    let negative = signed.then(|| {
        // LIVE at `bits == 64`, and the only check that catches a negative amount there. Noir types
        // a shift's amount as the _value's_ own type (`noir_tests/signed_shift` shifts an `i8` by
        // an `i8` and an `i32` by an `i32`), so on an `i64` the amount is an `i64` too (and
        // `cmp_bits` is then `64`, which makes the cast above an identity rather than a widening).
        // A negative amount therefore keeps its sign bit, and `below_width` does _not_ catch that:
        // `-1 s< 64` is true, so the bound test reports the shift as valid.
        //
        // Below 64 bits it is indeed dead, because the widening cast is a raw-bit zero-extension
        // (`Cast` masks, sign extension is the separate `SExt`) and so clears the sign bit at
        // `cmp_bits`. That is a fact about the narrow widths only and cannot be generalized into a
        // deletion of this check for now.
        let zero = emitter.int_const(cmp_bits, 0);
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
    let one_u1 = emitter.int_const(1, 1);
    emitter.emit(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: valid,
        rhs: one_u1,
    });
}
