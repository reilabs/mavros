//! The single definition of "this `Add`/`Sub`/`Mul` overflows".
//!
//! Noir rejects an overflowing `+`, `-` or `*` in constrained **and** unconstrained code:
//! `eval_constant_binary_op` reports `Failure` on a result that does not fit the width, and
//! `brillig_gen` emits an `add_overflow_check` ahead of the operation. Every mavros backend instead
//! _wraps_ for totality — the VM's `cell_add` masks, LLVM's `add` has no `nsw`, and `hlssa_to_r1cs`
//! folds with `wrapping_add` — so without this check R1CS and witness generation agree on the same
//! wrong answer and nothing downstream will notice.
//!
//! `LowerWitnessIntegerArithOps` computes the sum or product in the field and range-checks it back
//! down to the width, which rejects the same executions by construction rather than by an explicit
//! comparison. This module is the pure counterpart of that range check: same contract, different
//! machinery, because a pure operand reaches no constraint at all.

use crate::compiler::{
    analysis::value_range_analysis::ValueRange,
    ssa::{
        ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CastTarget, CmpKind, OpCode, Type, TypeExpr,
            assert_signed_op_width, builder::HLEmitter,
        },
    },
    util::bit_mask,
};

/// The width an `Add`/`Sub`/`Mul` is performed at, or `None` for an operand type whose arithmetic
/// cannot overflow.
///
/// This is the overflow counterpart of [`super::shift_guard::shift_operand_bits`], and returns the
/// width for the same reason: every predicate below is stated against it.
///
/// `Field` answers `None` deliberately, and is the one difference from
/// [`super::divmod_guard::divmod_can_fail`], which accepts it. Field arithmetic **wraps** rather
/// than failing.
pub fn overflow_operand_bits(lhs_type: &Type) -> Option<usize> {
    match lhs_type.strip_witness().expr {
        TypeExpr::Int(bits) => Some(bits),
        _ => None,
    }
}

/// Whether the range domain **proves** this operation stays inside its width, so the check
/// [`emit_overflow_cond`] would build is dead and need not be emitted at all.
///
/// This is the exact complement of that function: it discharges the same condition, computing the
/// operation's _mathematical_ result interval and asking whether the whole of it is representable.
/// Ordinary interval arithmetic is enough because `Add`, `Sub` and `Mul` are monotone in each
/// operand, so the endpoint hull is exact.
///
/// The reading is chosen by the operation, exactly as the check is: an unsigned `Add` overflows
/// when the raw sum leaves `[0, 2^bits)`, a signed one when the two's-complement sum leaves
/// `[−2^(bits−1), 2^(bits−1))`. Note that ⊥ answers `false`, like every other `proves_*` query.
///
/// This is a _constraint-eliding_ consumer of the domain, so it inherits the one assumption the
/// domain cannot discharge itself — the `WriteWitness` hint range that
/// [`super::divmod_guard::divmod_provably_defined`] spells out. The exposure is narrower here,
/// because every caller has already established that both operands are **pure**: in constrained
/// code that means they are compile-time constants, and in unconstrained code no constraint is
/// being elided at all, only a runtime assert.
pub fn overflow_provably_impossible(
    lhs: &ValueRange,
    rhs: &ValueRange,
    group: ArithGroup,
    bits: usize,
    signed: bool,
) -> bool {
    let (l, r) = if signed {
        (lhs.signed(), rhs.signed())
    } else {
        (lhs.unsigned(), rhs.unsigned())
    };

    let exact = match group {
        ArithGroup::Add => l.add(r),
        ArithGroup::Sub => l.sub(r),
        ArithGroup::Mul => l.mul(r),
        // A check that is not understood must not be discharged.
        _ => return false,
    };

    if signed {
        exact.proves_fits_in_signed_bits(bits)
    } else {
        exact.proves_fits_in_unsigned_bits(bits)
    }
}

/// A `u1` that is `1` exactly when `lhs op rhs` leaves the operand width.
///
/// `wrapped` is the operation's, the value it produces having wrapped, which is what the backends
/// compute and what the add/sub tests below compare against.
///
/// It is **required for `Add` and `Sub`, and ignored for `Mul`**, which decides which of the two
/// groups DCE may rewrite a dead operation into. `bits` is the operand width. A signed operation is
/// additionally bounded by [`assert_signed_op_width`], because the sign-bit and magnitude
/// arithmetic below is built at that width.
///
/// # Panics
///
/// If `wrapped` is `None` for an `Add` or a `Sub`.
pub fn emit_overflow_cond(
    emitter: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: ValueId,
    rhs: ValueId,
    wrapped: Option<ValueId>,
    bits: usize,
) -> ValueId {
    let signed = kind.is_signed();
    if signed {
        assert_signed_op_width(bits, "overflow check");
    }

    match (signed, kind.group()) {
        (_, group @ (ArithGroup::Add | ArithGroup::Sub)) => {
            let wrapped = wrapped.expect(
                "ICE: an add/sub overflow check compares against the operation's own result, so the caller must have emitted it; see `overflow_rewrite_saves_the_operation`",
            );
            if signed {
                signed_add_sub_overflow(emitter, group, lhs, rhs, wrapped, bits)
            } else {
                unsigned_add_sub_overflow(emitter, group, lhs, wrapped)
            }
        }
        (false, ArithGroup::Mul) => unsigned_mul_overflow(emitter, lhs, rhs, bits),
        (true, ArithGroup::Mul) => signed_mul_overflow(emitter, lhs, rhs, bits),
        (_, group) => unreachable!("overflow condition for a non-arithmetic group: {group:?}"),
    }
}

/// Emit `assert(!overflows(lhs, rhs))`, the unguarded form of the check.
///
/// The caller decides whether the operation itself accompanies it, and supplies `wrapped` on
/// [`emit_overflow_cond`]'s terms: `LowerPureGuards` emits the operation first and passes its
/// result, while `DCE` — which is deleting the operation — passes `None` and so may only do this
/// for a `Mul`.
///
/// # Panics
///
/// As [`emit_overflow_cond`] does, on a `None` `wrapped` for an `Add` or a `Sub`.
pub fn emit_no_overflow_assert(
    emitter: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: ValueId,
    rhs: ValueId,
    wrapped: Option<ValueId>,
    bits: usize,
) {
    let overflow = emit_overflow_cond(emitter, kind, lhs, rhs, wrapped, bits);
    let zero_u1 = emitter.int_const(1, 0);
    emitter.emit(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: overflow,
        rhs: zero_u1,
    });
}

/// The unsigned wrap test: the result moved the wrong way along the number line.
///
/// A sum that wraps lands _below_ the operand it was added to, and a difference that borrows lands
/// _above_ the one it was subtracted from; neither can happen without leaving the width. One
/// comparison covers every unsigned width because the wrapped result is already masked to it.
fn unsigned_add_sub_overflow(
    emitter: &mut impl HLEmitter,
    group: ArithGroup,
    lhs: ValueId,
    wrapped: ValueId,
) -> ValueId {
    match group {
        ArithGroup::Add => emitter.ult(wrapped, lhs),
        ArithGroup::Sub => emitter.ult(lhs, wrapped),
        other => unreachable!("unsigned add/sub overflow for {other:?}"),
    }
}

/// The signed wrap test, which is about **sign bits** rather than magnitude.
///
/// Two's-complement addition overflows exactly when the operands share a sign and the result does
/// not; subtraction exactly when they differ in sign and the result differs from the left operand.
/// Both are the same shape, which is why the two polarities share one instruction sequence: `Add`
/// wants the negation of `sign_l ^ sign_r` and `Sub` wants that value itself, so both are built and
/// DCE or SCS reclaims whichever went unused (measured to be true).
fn signed_add_sub_overflow(
    emitter: &mut impl HLEmitter,
    group: ArithGroup,
    lhs: ValueId,
    rhs: ValueId,
    wrapped: ValueId,
    bits: usize,
) -> ValueId {
    let sign_l = sign_bit(emitter, lhs, bits);
    let sign_r = sign_bit(emitter, rhs, bits);
    let sign_result = sign_bit(emitter, wrapped, bits);

    let signs_differ = emitter.xor(sign_l, sign_r);
    let signs_same = emitter.not(signs_differ);
    let sign_l_xor_result = emitter.xor(sign_l, sign_result);

    match group {
        ArithGroup::Add => emitter.and(signs_same, sign_l_xor_result),
        ArithGroup::Sub => emitter.and(signs_differ, sign_l_xor_result),
        other => unreachable!("signed add/sub overflow for {other:?}"),
    }
}

/// The unsigned multiply test, written **flat**: no branch, and one division.
///
/// `MAX / rhs < lhs` is the overflow condition for a nonzero `rhs`, and a multiply by zero cannot
/// overflow at all. The zero case is handled by *substituting* a safe divisor rather than by
/// branching on it, which is why this differs from the guarded lowering: a diamond would give
/// untaint, DCE and the block simplifiers three more blocks to chew on at every unguarded multiply
/// in the program for a condition that needs no control flow at all.
///
/// The substitution is `rhs | (rhs == 0)`, which is `rhs` untouched when it is nonzero and `1` when
/// it is not.
///
/// **No `rhs != 0` conjunct is needed**. Substituting `1` for a zero divisor makes the test
/// `MAX / 1 < lhs`, which is `MAX < lhs` — false for every `lhs`, since `lhs` is a `bits`-wide
/// value and `MAX` is the largest of those. So the substituted divisor answers "no overflow" on its
/// own, which is the right answer for a multiply by zero.
///
/// That argument depends on [`mul_overflows_nonzero`] using **`MAX` exactly** as its numerator. A
/// smaller bound there would make `1` an unsafe substitute and the conjunct would have to come
/// back.
fn unsigned_mul_overflow(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    bits: usize,
) -> ValueId {
    let safe_rhs = substitute_one_for_zero(emitter, rhs, bits);
    mul_overflows_nonzero(emitter, lhs, safe_rhs, bits)
}

/// `value | (value == 0)`: the value itself, or `1` where it is zero.
///
/// The branch-free stand-in for a divisor that must not be zero. See [`unsigned_mul_overflow`] for
/// why it is an `Or` rather than a `Select` or an `Add`.
fn substitute_one_for_zero(emitter: &mut impl HLEmitter, value: ValueId, bits: usize) -> ValueId {
    let zero = emitter.int_const(bits, 0);
    let is_zero = emitter.eq(value, zero);
    let bump = emitter.cast_to(CastTarget::Int(bits), is_zero);
    emitter.or(value, bump)
}

/// `MAX / rhs < lhs`: the unsigned multiply overflow test for a divisor known to be nonzero.
///
/// Shared by the flat form above and by `LowerPureGuards`' guarded lowering, which reaches it on
/// the arm where it has already branched on `rhs != 0`. Keeping the two on one predicate is the
/// point of this module.
///
/// The division is exact only downwards, which is what makes this the right comparison: `MAX / rhs`
/// floors, so it is the largest `lhs` whose product still fits, and the test is a strict `<`.
///
/// The numerator being **`MAX` exactly** is load-bearing beyond this function: it is what lets
/// [`unsigned_mul_overflow`] substitute `1` for a zero divisor and drop the `rhs != 0` conjunct.
pub fn mul_overflows_nonzero(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    bits: usize,
) -> ValueId {
    let max = emitter.int_const(bits, bit_mask(bits));
    let limit = emitter.udiv(max, rhs);
    emitter.ult(limit, lhs)
}

/// The signed multiply test, flat for the reason [`unsigned_mul_overflow`] is.
///
/// The product's sign is known before the product is: it is `sign(lhs) ^ sign(rhs)`. That matters
/// because the signed range is **asymmetric** — `−2^(bits−1)` is representable and `+2^(bits−1)` is
/// not — so the magnitude bound depends on which way the result points.
///
/// The zero divisor falls out as it does in [`unsigned_mul_overflow`] for the same reason.
/// `|rhs| == 0` only when `rhs == 0`, so `rhs` is then non-negative, the product's sign is `lhs`'s,
/// and the bound `MAX_S + sign(lhs)` is exactly the largest magnitude an operand of that sign can
/// have — `MAX_S` for a positive `lhs`, `2^(bits−1)` for a negative one, which is `INT_MIN`'s. So
/// `bound / 1 < |lhs|` is false whatever `lhs` is, which is the right answer for a multiply by
/// zero.
fn signed_mul_overflow(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    bits: usize,
) -> ValueId {
    let operands = signed_mul_operands(emitter, lhs, rhs, bits);
    let safe_abs_rhs = substitute_one_for_zero(emitter, operands.abs_rhs, bits);
    signed_mul_magnitude_overflows(
        emitter,
        operands.abs_lhs,
        safe_abs_rhs,
        operands.result_sign,
        bits,
    )
}

/// What [`signed_mul_magnitude_overflows`] needs, decomposed out of a signed multiply's operands.
pub struct SignedMulOperands {
    /// `|lhs|`, as an unsigned integer of the operand width.
    pub abs_lhs: ValueId,

    /// `|rhs|`, as an unsigned integer of the operand width.
    pub abs_rhs: ValueId,

    /// The product's sign as a `u1`, known before the product is.
    pub result_sign: ValueId,
}

/// Split a signed multiply into the magnitudes and result sign its overflow test is stated over.
pub fn signed_mul_operands(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    bits: usize,
) -> SignedMulOperands {
    let sign_lhs = sign_bit(emitter, lhs, bits);
    let sign_rhs = sign_bit(emitter, rhs, bits);
    let result_sign = emitter.xor(sign_lhs, sign_rhs);
    let abs_lhs = abs_as_u(emitter, lhs, sign_lhs, bits);
    let abs_rhs = abs_as_u(emitter, rhs, sign_rhs, bits);

    SignedMulOperands {
        abs_lhs,
        abs_rhs,
        result_sign,
    }
}

/// `(MAX_S + result_sign) / |rhs| < |lhs|`: the signed multiply overflow test for a magnitude known
/// to be nonzero.
///
/// The counterpart of [`mul_overflows_nonzero`], shared with `LowerPureGuards`' guarded lowering on
/// the same terms. `result_sign` shifts the bound from `2^(bits−1) − 1` to `2^(bits−1)`, which is
/// the one extra magnitude a negative product may have.
///
/// Both operands are absolute values under unsigned opcodes; only the product itself is signed.
pub fn signed_mul_magnitude_overflows(
    emitter: &mut impl HLEmitter,
    abs_lhs: ValueId,
    abs_rhs: ValueId,
    result_sign: ValueId,
    bits: usize,
) -> ValueId {
    let positive_max = emitter.int_const(bits, (1u128 << (bits - 1)) - 1);
    let result_sign = emitter.cast_to(CastTarget::Int(bits), result_sign);
    let max_magnitude = emitter.uadd(positive_max, result_sign);
    let limit = emitter.udiv(max_magnitude, abs_rhs);
    emitter.ult(limit, abs_lhs)
}

/// The value's sign bit, as a `u1`.
pub fn sign_bit(emitter: &mut impl HLEmitter, value: ValueId, bits: usize) -> ValueId {
    let sign = emitter.bit_range(value, bits - 1, 1);
    emitter.cast_to(CastTarget::Int(1), sign)
}

/// The value's magnitude, as an unsigned integer of the same width.
///
/// Computed in the field rather than with a negation, because the one value this has to get right
/// is `INT_MIN`, whose magnitude is not representable at the width: `−(−2^(bits−1))` wraps back to
/// itself. Lifting to the field first means `2^(bits−1)` is an ordinary number until the final cast
/// puts it back, where it is the one pattern whose unsigned reading is what the magnitude needs.
///
/// `sign_u1` is the value's own sign bit, which the caller already has.
pub fn abs_as_u(
    emitter: &mut impl HLEmitter,
    value: ValueId,
    sign_u1: ValueId,
    bits: usize,
) -> ValueId {
    // FIELD-ASSUMPTION: L4-modulus-query. `two_pow(bits)` below is the value it is meant to be only
    // while it has not wrapped, and if it has, the magnitude is silently wrong and so is every
    // overflow test built on it — a wrong answer rather than a rejection. On bn254 the 64-bit
    // signed cap leaves ample room; on a narrower field this refuses instead. Same construction and
    // same bound as `LowerWitnessBitwiseOps::lower_integer_sext`, which lifts a value into the
    // field the same way.
    assert!(
        bits < emitter.field().field_bit_size() as usize,
        "a {bits}-bit magnitude needs a field wider than {} bits",
        emitter.field().field_bit_size()
    );

    let value_field = emitter.cast_to_field(value);
    let sign = emitter.cast_to_field(sign_u1);
    let sign_shift = emitter.field_const(emitter.field().two_pow(bits));
    let sign_shifted = emitter.umul(sign, sign_shift);
    let signed_value = emitter.usub(value_field, sign_shifted);
    let two = emitter.field_const(emitter.field().constant(2));
    let two_sign = emitter.umul(two, sign);
    let one = emitter.field_const(emitter.field().constant(1));
    let factor = emitter.usub(one, two_sign);
    let abs = emitter.umul(signed_value, factor);
    emitter.cast_to(CastTarget::Int(bits), abs)
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::analysis::value_range_analysis::{Interval, Width};

    fn u_range(bits: usize, lo: i64, hi: i64) -> ValueRange {
        ValueRange::from_unsigned(Width::Bits(bits), Interval::closed(lo, hi))
    }

    fn s_range(bits: usize, lo: i64, hi: i64) -> ValueRange {
        ValueRange::from_signed(Width::Bits(bits), Interval::closed(lo, hi))
    }

    #[test]
    fn an_unsigned_sum_discharges_exactly_at_the_top_of_the_width() {
        // 200 + 55 == 255 fits a `u8`; one more does not.
        assert!(overflow_provably_impossible(
            &u_range(8, 200, 200),
            &u_range(8, 0, 55),
            ArithGroup::Add,
            8,
            false
        ));
        assert!(!overflow_provably_impossible(
            &u_range(8, 200, 200),
            &u_range(8, 0, 56),
            ArithGroup::Add,
            8,
            false
        ));
    }

    #[test]
    fn an_unsigned_difference_needs_the_left_operand_to_dominate() {
        assert!(overflow_provably_impossible(
            &u_range(8, 10, 20),
            &u_range(8, 0, 10),
            ArithGroup::Sub,
            8,
            false
        ));
        // The ranges overlap, so `lhs - rhs` can borrow.
        assert!(!overflow_provably_impossible(
            &u_range(8, 10, 20),
            &u_range(8, 0, 11),
            ArithGroup::Sub,
            8,
            false
        ));
    }

    #[test]
    fn a_product_is_bounded_by_the_endpoint_hull() {
        assert!(overflow_provably_impossible(
            &u_range(8, 0, 15),
            &u_range(8, 0, 17),
            ArithGroup::Mul,
            8,
            false
        ));
        assert!(!overflow_provably_impossible(
            &u_range(8, 0, 16),
            &u_range(8, 0, 16),
            ArithGroup::Mul,
            8,
            false
        ));
    }

    #[test]
    fn the_sign_parameter_decides_which_question_is_asked() {
        // The same pair of operands is a legal `u8` sum and an overflowing `i8` one: 100 + 100 is
        // 200, which fits `[0, 256)` and does not fit `[-128, 128)`.
        let hundred = u_range(8, 100, 100);
        assert!(overflow_provably_impossible(
            &hundred,
            &hundred,
            ArithGroup::Add,
            8,
            false
        ));

        let hundred_s = s_range(8, 100, 100);
        assert!(!overflow_provably_impossible(
            &hundred_s,
            &hundred_s,
            ArithGroup::Add,
            8,
            true
        ));
    }

    #[test]
    fn a_signed_difference_can_overflow_downwards() {
        // `-100 - 100` is -200, below `i8`'s floor, so this is the mirror of the case above and
        // shows the discharge is not merely testing magnitude.
        assert!(!overflow_provably_impossible(
            &s_range(8, -100, -100),
            &s_range(8, 100, 100),
            ArithGroup::Sub,
            8,
            true
        ));
        assert!(overflow_provably_impossible(
            &s_range(8, -100, -100),
            &s_range(8, 0, 28),
            ArithGroup::Sub,
            8,
            true
        ));
    }

    #[test]
    fn bottom_never_discharges_the_check() {
        // ⊥ is derived _from_ the constraints the check is part of, so reading it as "cannot
        // overflow" is circular. See the proof-strength predicates in `value_range_analysis`.
        let bottom = ValueRange::empty(Width::Bits(8));
        let small = u_range(8, 0, 1);
        for group in [ArithGroup::Add, ArithGroup::Sub, ArithGroup::Mul] {
            assert!(!overflow_provably_impossible(
                &bottom, &small, group, 8, false
            ));
            assert!(!overflow_provably_impossible(
                &small, &bottom, group, 8, false
            ));
        }
    }

    #[test]
    fn an_unbounded_operand_never_discharges_the_check() {
        // `Width::NonScalar` is what a value with no range information gets, and its interval is
        // top. An unbounded endpoint must not be read as a bound.
        let unknown = ValueRange::full(Width::NonScalar);
        let small = u_range(8, 0, 1);
        assert!(!overflow_provably_impossible(
            &unknown,
            &small,
            ArithGroup::Add,
            8,
            false
        ));
    }

    #[test]
    fn a_group_without_an_overflow_condition_is_never_discharged() {
        // `overflow_operand_bits` and the calling arms keep these away, but answering "provably
        // impossible" for a group this does not understand would silently delete a check if the two
        // ever drifted apart. The operands here are ones that _would_ discharge an add.
        let small = u_range(8, 0, 1);
        for group in [
            ArithGroup::Div,
            ArithGroup::Rem,
            ArithGroup::Shl,
            ArithGroup::Shr,
            ArithGroup::And,
            ArithGroup::Or,
            ArithGroup::Xor,
        ] {
            assert!(!overflow_provably_impossible(
                &small, &small, group, 8, false
            ));
        }
    }

    /// Build a block emitter over a throwaway function and hand it to `body`.
    fn with_emitter(
        body: impl FnOnce(&mut crate::compiler::ssa::hlssa::builder::HLBlockEmitter<'_>),
    ) {
        use crate::compiler::ssa::hlssa::{HLSSA, builder::HLSSABuilder};

        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            body(&mut e);
            e.terminate_return(vec![]);
        });
    }

    #[test]
    #[should_panic(expected = "the caller must have emitted it")]
    fn an_add_check_without_the_sum_is_a_compiler_bug() {
        // The add/sub test is a comparison against the operation's own result, so there is nothing
        // to build it from when the caller is deleting the operation. Rebuilding the sum here
        // would resurrect what `DCE` came to remove, so `None` is refused rather than filled in --
        // and `overflow_rewrite_saves_the_operation` is what keeps `DCE` off this path.
        with_emitter(|e| {
            let a = e.int_const(8, 1);
            let b = e.int_const(8, 2);
            emit_overflow_cond(e, BinaryArithOpKind::UAdd, a, b, None, 8);
        });
    }

    #[test]
    fn a_multiply_check_needs_no_sum() {
        // The other half of the same decision: a multiply's test is built from the operands alone,
        // which is exactly why it is the group `DCE` may rewrite a dead operation into.
        with_emitter(|e| {
            let a = e.int_const(8, 1);
            let b = e.int_const(8, 2);
            emit_overflow_cond(e, BinaryArithOpKind::UMul, a, b, None, 8);
        });
    }

    #[test]
    fn only_an_integer_operand_can_overflow() {
        assert_eq!(overflow_operand_bits(&Type::int(32)), Some(32));
        // Field arithmetic wraps rather than failing, so there is nothing to reject.
        assert_eq!(overflow_operand_bits(&Type::field()), None);
        assert_eq!(overflow_operand_bits(&Type::function()), None);
    }
}

/// The discharge predicate's conformance relation to the normative model in `mavros-int-semantics`.
///
/// [`overflow_provably_impossible`] is the one place in the guard IR that answers a question whose
/// wrong answer is *silent*: every other function here builds a check, and a mistake in one of them
/// rejects a program it should not, which a test corpus notices. This one **deletes** a check, and
/// a mistake in it produces a proof for a program Noir rejects.
///
/// So the relation is the one-sided one that failure mode names:
///
/// > if `overflow_provably_impossible(L, R, ..)` then for every `a ∈ γ(L)` and `b ∈ γ(R)`,
/// > [`eval`](mavros_int_semantics::eval) does not reject `a op b`.
///
/// Nothing is claimed in the other direction. Declining to discharge is always safe, so a range
/// pair this refuses is not required to be one that can overflow — that is the difference between
/// a check being *needed* and a check being *emitted*, and only the first is a correctness
/// question.
///
/// γ is over **both** readings, matching `value_range_analysis`'s own sweep: a `ValueRange` denotes
/// the patterns its unsigned and signed intervals both admit, so enumerating one reading would feed
/// the model operands the range does not actually contain.
#[cfg(test)]
mod int_semantics_conformance {
    use super::*;
    use crate::compiler::analysis::value_range_analysis::{Interval, Width};
    use mavros_int_semantics::{self as semantics, IntBits, IntOp, Outcome, corners};
    use num_bigint::BigInt;

    /// Whether a bit pattern is in γ of a range: admitted by both readings.
    fn in_gamma(range: &ValueRange, bits: usize, v: u128) -> bool {
        range.unsigned().contains(&BigInt::from(v))
            && range
                .signed()
                .contains(&IntBits::from_u128(bits, v).to_signed())
    }

    /// Every bit pattern a range denotes, at a width small enough to enumerate.
    fn gamma(range: &ValueRange, bits: usize) -> Vec<u128> {
        (0..=semantics::mask(bits))
            .filter(|v| in_gamma(range, bits, *v))
            .collect()
    }

    /// The ranges each operand is swept over at `bits`.
    ///
    /// Chosen for what they straddle rather than for coverage, as `value_range_analysis`'s sweep
    /// is: the whole width, the singletons at each end, runs sitting just inside each boundary and
    /// astride the sign one, and two entered through the *signed* reading so that the reduction is
    /// exercised from both sides.
    fn input_ranges(bits: usize) -> Vec<ValueRange> {
        let width = Width::Bits(bits);
        let top = semantics::mask(bits);
        let half = top / 2;
        let closed = |lo: u128, hi: u128| {
            ValueRange::from_unsigned(width, Interval::closed(BigInt::from(lo), BigInt::from(hi)))
        };
        let signed = |lo: i64, hi: i64| {
            ValueRange::from_signed(width, Interval::closed(BigInt::from(lo), BigInt::from(hi)))
        };

        vec![
            ValueRange::full(width),
            closed(0, 0),
            closed(top, top),
            closed(0, top.min(3)),
            closed(1, top.min(5)),
            closed(half, top.min(half + 3)),
            closed(half.saturating_sub(2), top.min(half + 2)),
            signed(-1, 1),
            signed(-3, 3),
        ]
    }

    /// Discharging a check never deletes a rejection the model requires.
    #[test]
    fn a_discharged_check_is_one_no_input_could_have_failed() {
        let mut discharged = 0usize;

        for bits in corners::EXHAUSTIVE_WIDTHS {
            for group in [ArithGroup::Add, ArithGroup::Sub, ArithGroup::Mul] {
                for signed in [false, true] {
                    let op = IntOp::from(BinaryArithOpKind::with_sign(group, signed));
                    for l in &input_ranges(bits) {
                        for r in &input_ranges(bits) {
                            if !overflow_provably_impossible(l, r, group, bits, signed) {
                                continue;
                            }
                            discharged += 1;

                            for a in gamma(l, bits) {
                                for b in gamma(r, bits) {
                                    let outcome = semantics::eval(
                                        op,
                                        &IntBits::from_u128(bits, a),
                                        &IntBits::from_u128(bits, b),
                                    );
                                    assert!(
                                        !matches!(outcome, Outcome::Rejected(_)),
                                        "{op:?} at {bits} bits discharged the check \
                                         for {l:?} and {r:?}, but {a:#x} {b:#x} is {outcome:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // A predicate that discharged nothing would satisfy every assertion above and elide no
        // check at all, so the count is part of the test rather than a diagnostic.
        assert!(
            discharged > 100,
            "the sweep only reached {discharged} discharged checks"
        );
    }
}
