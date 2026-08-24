//! Tests of the semantics engendered by the model.
//!
//! It encompasses three kinds of test:
//!
//! 1. The **Pins** are hand-written cases taken straight from Noir's own tests and documentation.
//!    These ensure that the model matches Noir's semantics, rather than simply being internally
//!    consistent.
//! 2. The **Exhaustive** tests check every operand pair at `bits ∈ {1, 4, 8}` in both readings for
//!    all ten operations. Structural bugs (masking, sign decode, boundary-off-by-one) are generic
//!    across widths, so an exhaustive sweep here will highlight them.
//! 3. The **Property** tests run over the rest of the domain to provide extra coverage where the
//!    exhaustive tier cannot.

use proptest::prelude::*;

use crate::{
    CmpOp, FIELD_LIMB_BITS, IntOp, MAX_BITS, MAX_SIGNED_BITS, Outcome, Raw, Reject, Sign,
    bit_range, cast_int, cmp, corners, decode_signed, encode_signed, eval, field_limbs_fit,
    field_limbs_to_int, fits_signed, limbs_to_int, mask, masked_shift_amount, not, residue,
    sign_extend, signed_max, signed_min,
};

// PROPTEST UTILITIES
// ================================================================================================

/// Shorthand for the common equal-width call.
fn ev(op: IntOp, sign: Sign, bits: usize, lhs: Raw, rhs: Raw) -> Outcome {
    eval(op, sign, bits, lhs, bits, rhs)
}

// LAYER 1: PINS AGAINST NOIR
// ================================================================================================

#[test]
fn arithmetic_overflow_is_rejected_not_wrapped() {
    // `eval_constant_binary_op` computes the result and then rejects it when `result.ilog2() >=
    // bit_size`. This is the rule mavros currently breaks for unguarded pure operations: every
    // backend wraps to 44 and the program runs clean.
    assert_eq!(
        ev(IntOp::Add, Sign::Unsigned, 8, 200, 100),
        Outcome::Rejected(Reject::Overflow)
    );
    // ... and the residue records what those backends compute, so a conformance test can hold them
    // to it without pretending it is the right answer.
    assert_eq!(
        residue(IntOp::Add, Sign::Unsigned, 8, 200, 8, 100),
        Some(44)
    );

    // Unsigned subtraction below zero is an overflow, not a wrap to a large number.
    assert_eq!(
        ev(IntOp::Sub, Sign::Unsigned, 8, 1, 2),
        Outcome::Rejected(Reject::Overflow)
    );
    assert_eq!(residue(IntOp::Sub, Sign::Unsigned, 8, 1, 8, 2), Some(255));

    assert_eq!(
        ev(IntOp::Mul, Sign::Unsigned, 8, 16, 16),
        Outcome::Rejected(Reject::Overflow)
    );

    // Signed overflow is the same story at the signed boundary.
    let max = encode_signed(8, signed_max(8));
    assert_eq!(
        ev(IntOp::Add, Sign::Signed, 8, max, 1),
        Outcome::Rejected(Reject::Overflow)
    );
    assert_eq!(
        residue(IntOp::Add, Sign::Signed, 8, max, 8, 1),
        Some(encode_signed(8, signed_min(8))),
        "127i8 + 1 wraps to -128 in every backend"
    );

    // In range, so accepted, and the two readings differ on the same patterns.
    assert_eq!(
        ev(IntOp::Add, Sign::Unsigned, 8, 200, 55),
        Outcome::Value(255)
    );
    assert_eq!(
        ev(IntOp::Add, Sign::Signed, 8, 200, 55),
        Outcome::Value(255)
    );
}

#[test]
fn division_rejects_a_zero_divisor_and_the_signed_overflow() {
    for sign in Sign::ALL {
        for op in [IntOp::Div, IntOp::Rem] {
            assert_eq!(ev(op, sign, 8, 7, 0), Outcome::Rejected(Reject::DivByZero));
            assert_eq!(
                residue(op, sign, 8, 7, 8, 0),
                None,
                "no agreed answer exists"
            );
        }
    }

    // `INT_MIN / -1` has a quotient one past the top of the type. `expand_signed_math` emits an
    // explicit constrain for exactly this pair.
    let min = encode_signed(8, signed_min(8));
    let minus_one = encode_signed(8, -1);
    assert_eq!(
        ev(IntOp::Div, Sign::Signed, 8, min, minus_one),
        Outcome::Rejected(Reject::DivOverflow)
    );
    // `INT_MIN % -1` would produce 0, which is representable -- but Noir defines the remainder in
    // terms of the same quotient, so it rejects that too.
    assert_eq!(
        ev(IntOp::Rem, Sign::Signed, 8, min, minus_one),
        Outcome::Rejected(Reject::DivOverflow)
    );
    // The unsigned reading of the same patterns is 128 / 255, which is perfectly fine.
    assert_eq!(
        ev(IntOp::Div, Sign::Unsigned, 8, min, minus_one),
        Outcome::Value(0)
    );
}

#[test]
fn signed_division_truncates_toward_zero() {
    // `noir_tests/signed_divmod` pins this, and it is what separates `/` from an arithmetic `>>`:
    // `-7 / 2` is -3 while `-7 >> 1` is -4.
    let neg7 = encode_signed(8, -7);
    assert_eq!(
        ev(IntOp::Div, Sign::Signed, 8, neg7, 2).value(),
        Some(encode_signed(8, -3))
    );
    assert_eq!(
        ev(IntOp::Rem, Sign::Signed, 8, neg7, 2).value(),
        Some(encode_signed(8, -1))
    );
    assert_eq!(
        ev(IntOp::Shr, Sign::Signed, 8, neg7, 1).value(),
        Some(encode_signed(8, -4))
    );

    // The remainder follows the dividend, not the divisor.
    let neg2 = encode_signed(8, -2);
    assert_eq!(
        ev(IntOp::Div, Sign::Signed, 8, 7, neg2).value(),
        Some(encode_signed(8, -3))
    );
    assert_eq!(ev(IntOp::Rem, Sign::Signed, 8, 7, neg2).value(), Some(1));
}

#[test]
fn a_left_shift_wraps_and_only_the_amount_can_reject() {
    // `noir_tests/specialized_shl_wrap` exists to pin this: 200 << 1 is 400, which keeps only its
    // low eight bits, so the answer is 144 and *not* a rejection. Losing bits off the top is the
    // specified behaviour of `<<`, unlike `*`, where it is an error.
    assert_eq!(
        ev(IntOp::Shl, Sign::Unsigned, 8, 200, 1),
        Outcome::Value(144)
    );
    assert_eq!(
        ev(IntOp::Mul, Sign::Unsigned, 8, 200, 2),
        Outcome::Rejected(Reject::Overflow)
    );

    // A signed `<<` wraps across the sign boundary: 64i8 << 1 is -128.
    assert_eq!(
        ev(IntOp::Shl, Sign::Signed, 8, 64, 1).value(),
        Some(encode_signed(8, signed_min(8)))
    );
    // ... which is why the model shifts the raw pattern rather than asking whether `64 * 2` fits.
    assert_eq!(ev(IntOp::Shl, Sign::Signed, 8, 1, 7).value(), Some(0x80));
}

#[test]
fn a_shift_amount_at_or_above_the_width_is_rejected() {
    // `remove_bit_shifts`' module doc: "In all cases, if the shift amount is equal to or exceeds
    // the operand's number of bits, the result will be a constrain failure". The boundary is the
    // whole content of the rule, so both sides of it are pinned.
    for op in [IntOp::Shl, IntOp::Shr] {
        for sign in Sign::ALL {
            assert!(
                ev(op, sign, 8, 1, 7).value().is_some(),
                "7 is the largest legal amount"
            );
            assert_eq!(
                ev(op, sign, 8, 1, 8),
                Outcome::Rejected(Reject::ShiftAmount)
            );
            assert_eq!(
                ev(op, sign, 8, 1, 100),
                Outcome::Rejected(Reject::ShiftAmount)
            );
        }
    }

    // Noir's own `shl_overflow_u64` / `shr_overflow_u64` are a u64 shifted by 100.
    for op in [IntOp::Shl, IntOp::Shr] {
        assert_eq!(
            ev(op, Sign::Unsigned, 64, 1, 100),
            Outcome::Rejected(Reject::ShiftAmount)
        );
    }

    // A *negative* amount is the same rejection and not a separate rule:
    // `enforce_bitshift_rhs_lt_bit_size` casts the amount to unsigned first, so -1 reads as a huge
    // magnitude. `shl_signed_regression_9592` is this case.
    let minus_one = encode_signed(32, -1);
    assert_eq!(
        ev(IntOp::Shl, Sign::Signed, 32, 1, minus_one),
        Outcome::Rejected(Reject::ShiftAmount)
    );
}

#[test]
fn a_right_shift_fills_by_its_reading() {
    // Unsigned zero-fills, signed sign-fills and saturates at -1 rather than reaching zero.
    assert_eq!(
        ev(IntOp::Shr, Sign::Unsigned, 8, 0xF0, 2),
        Outcome::Value(0x3C)
    );
    assert_eq!(
        ev(IntOp::Shr, Sign::Signed, 8, 0xF0, 2),
        Outcome::Value(0xFC)
    );
    assert_eq!(
        ev(IntOp::Shr, Sign::Signed, 8, 0xF0, 7).value(),
        Some(encode_signed(8, -1))
    );
    assert_eq!(ev(IntOp::Shr, Sign::Signed, 8, 0, 7), Outcome::Value(0));
}

#[test]
fn bitwise_operations_have_no_reading_and_never_reject() {
    for sign in Sign::ALL {
        assert_eq!(ev(IntOp::And, sign, 8, 0xF0, 0x3C), Outcome::Value(0x30));
        assert_eq!(ev(IntOp::Or, sign, 8, 0xF0, 0x3C), Outcome::Value(0xFC));
        assert_eq!(ev(IntOp::Xor, sign, 8, 0xF0, 0x3C), Outcome::Value(0xCC));
    }
}

#[test]
fn comparison_reads_the_patterns_as_the_operation_says() {
    // 0xFB is 251 unsigned and -5 signed, so the two readings order it against 2 differently.
    assert!(cmp(CmpOp::Lt, Sign::Signed, 8, 0xFB, 2));
    assert!(!cmp(CmpOp::Lt, Sign::Unsigned, 8, 0xFB, 2));
    // Equality needs no reading at all.
    for sign in Sign::ALL {
        assert!(cmp(CmpOp::Eq, sign, 8, 0xFB, 0xFB));
        assert!(!cmp(CmpOp::Eq, sign, 8, 0xFB, 2));
    }
}

#[test]
fn the_field_cast_reads_both_low_limbs() {
    // The bug this exists to prevent: reading only `limbs[0]` answers a different number for every
    // `Field as u128` above 2^64, which is exactly the range `u128` was added for.
    let limbs = [7u64, 1u64, 0, 0];
    assert_eq!(field_limbs_to_int(&limbs, 128), (1u128 << 64) | 7);
    assert_eq!(
        field_limbs_to_int(&limbs, 64),
        7,
        "narrowing still truncates"
    );
    assert_eq!(field_limbs_to_int(&limbs, 8), 7);
}

#[test]
fn the_field_cast_is_indifferent_to_the_limb_count() {
    // bn254 has four limbs and a smaller field has fewer, so the count is the field's business and
    // not this crate's. Every one of these describes the same element.
    assert_eq!(field_limbs_to_int(&[7u64], 128), 7);
    assert_eq!(field_limbs_to_int(&[7u64, 0, 0], 128), 7);
    assert_eq!(field_limbs_to_int(&[7u64, 0, 0, 0, 0, 0], 128), 7);

    // Limbs past the second hold only bits `mask` discards, so a longer slice cannot change the
    // answer -- the guard against a future width picking them up silently.
    assert_eq!(
        field_limbs_to_int(&[7u64, 1, u64::MAX, u64::MAX], 128),
        (1u128 << 64) | 7
    );
    assert_eq!(
        field_limbs_to_int(&[], 128),
        0,
        "no limbs is the zero element"
    );
}

#[test]
fn the_fit_test_covers_exactly_the_bits_the_read_keeps() {
    // The companion of the truncating read above: whatever `field_limbs_fit` accepts,
    // `field_limbs_to_int` must have lost nothing from.
    assert!(field_limbs_fit(&[7u64, 1, 0, 0], 128));
    assert!(!field_limbs_fit(&[7u64, 1, 1, 0], 128));
    assert!(!field_limbs_fit(&[0u64, 0, 0, 1], 128));
    assert!(field_limbs_fit(&[], 128), "no limbs is the zero element");

    // A width that does not end on a limb boundary is the case the obvious spelling gets wrong:
    // `70 / 64` is `1`, so "every limb from index 1 up is zero" would reject an element whose
    // seventieth bit is the highest one set, which fits perfectly well.
    assert!(field_limbs_fit(&[u64::MAX, 0b11_1111, 0, 0], 70));
    assert!(!field_limbs_fit(&[u64::MAX, 0b111_1111, 0, 0], 70));

    // And the boundary width itself, where the limb above must be empty outright.
    assert!(field_limbs_fit(&[u64::MAX, 0, 0, 0], 64));
    assert!(!field_limbs_fit(&[u64::MAX, 1, 0, 0], 64));

    // The relation the two entry points hold, swept over the corners: accepting means the read is
    // lossless, which is the only reason a caller asks.
    for bits in [1usize, 8, 63, 64, 65, 70, 127, 128] {
        for limbs in [
            [0u64, 0, 0, 0],
            [1, 0, 0, 0],
            [u64::MAX, 0, 0, 0],
            [u64::MAX, u64::MAX, 0, 0],
            [0, 0, 1, 0],
            [7, 1, 0, 0],
        ] {
            if !field_limbs_fit(&limbs, bits) {
                continue;
            }
            assert_eq!(
                field_limbs_to_int(&limbs, bits),
                field_limbs_to_int(&limbs, MAX_BITS),
                "{limbs:?} was said to fit {bits} bits but reading it there lost something"
            );
        }
    }
}

#[test]
fn limb_width_is_a_parameter_not_an_assumption() {
    // A canonical field limb is 64 bits wide whatever the field, so the field-shaped entry point is
    // exactly the general one pinned at that width.
    let canonical = [7u64, 1, 0, 0];
    assert_eq!(
        limbs_to_int(&canonical, FIELD_LIMB_BITS, 128),
        field_limbs_to_int(&canonical, 128)
    );

    // The witness decompositions in `witness_bitwise` use a limb width derived from the field size,
    // so the same little-endian vector says something different at a different width.
    assert_eq!(limbs_to_int(&[7u64, 1, 0, 0], 32, 128), (1u128 << 32) | 7);
    assert_eq!(limbs_to_int(&[1u64, 0, 1], 1, 128), 0b101);

    // A narrow limb packed into a wider container carries junk above its width, which is the
    // caller's normal state rather than an error.
    assert_eq!(limbs_to_int(&[u64::MAX], 32, 128), u128::from(u32::MAX));

    // A limb at least as wide as the result needs no recombination at all.
    assert_eq!(limbs_to_int(&[7u64], MAX_BITS, 128), 7);
}

#[test]
#[should_panic(expected = "a limb must be at least one bit wide")]
fn a_zero_width_limb_describes_no_representation() {
    let _ = limbs_to_int(&[7u64], 0, 128);
}

#[test]
fn bit_level_helpers_agree_with_their_definitions() {
    assert_eq!(cast_int(0x1FF, 8), 0xFF);
    assert_eq!(
        cast_int(0xFF, 32),
        0xFF,
        "widening a raw pattern zero-extends"
    );
    assert_eq!(
        sign_extend(0xFF, 8, 32),
        0xFFFF_FFFF,
        "widening a signed one does not"
    );
    assert_eq!(sign_extend(0x7F, 8, 32), 0x7F);
    assert_eq!(bit_range(0xABCD, 8, 8), 0xAB);
    assert_eq!(bit_range(0xABCD, 0, 8), 0xCD);
    assert_eq!(not(0xF0, 8), 0x0F);
    assert_eq!(decode_signed(8, 0xFF), -1);
    assert_eq!(
        decode_signed(1, 1),
        -1,
        "the only negative a one-bit pattern holds"
    );
    assert_eq!(encode_signed(8, -1), 0xFF);
}

#[test]
fn the_shift_backstop_masks_to_the_operand_width_not_the_host() {
    // The clause that catches an evaluator inheriting a host width: `u128::wrapping_shl` masks to
    // 127, so an evaluator built on it answers `1u8 << 8 == 0` where the VM and LLVM, which mask
    // to `bits - 1`, answer `1`. `hlssa_to_r1cs::arith` is built on it today.
    assert_eq!(masked_shift_amount(8, 8), 0);
    assert_eq!(masked_shift_amount(100, 64), 36);
    assert_eq!(residue(IntOp::Shl, Sign::Unsigned, 8, 1, 8, 8), Some(1));
    assert_ne!(residue(IntOp::Shl, Sign::Unsigned, 8, 1, 8, 8), Some(0));

    // At a non-power-of-two width the mask is a submask rather than a modulo -- 9 & 6 is 0, not
    // 9 % 7 == 2 -- and both backends compute that same submask, so the model records it.
    assert_eq!(masked_shift_amount(9, 7), 0);
    assert!(masked_shift_amount(u128::MAX, 7) < 7);
}

// LAYER 2: EXHAUSTIVE
// ================================================================================================

/// The invariant that makes the two-function split safe: a residue never contradicts an accepted
/// evaluation, it only extends it into the rejected region.
fn check_point(op: IntOp, sign: Sign, bits: usize, lhs: Raw, rhs_bits: usize, rhs: Raw) {
    let outcome = eval(op, sign, bits, lhs, rhs_bits, rhs);
    let residue = residue(op, sign, bits, lhs, rhs_bits, rhs);

    if let Outcome::Value(v) = outcome {
        assert!(
            v <= mask(bits),
            "{op:?}/{sign:?}/{bits} produced {v:#x}, outside its width, from {lhs:#x} {rhs:#x}"
        );
        assert_eq!(
            residue,
            Some(v),
            "{op:?}/{sign:?}/{bits} residue contradicts its own accepted value"
        );
        if sign.is_signed() {
            assert!(
                fits_signed(bits, decode_signed(bits, v)),
                "{op:?}/{bits} produced a pattern that does not decode into its own width"
            );
        }
    }

    if let Some(r) = residue {
        assert!(
            r <= mask(bits),
            "{op:?}/{sign:?}/{bits} residue {r:#x} escaped its width"
        );
    }

    // `None` is reserved for the two division cases and nothing else; a new `None` would mean the
    // backends had quietly stopped agreeing somewhere the model has not noticed.
    if residue.is_none() {
        assert!(
            matches!(
                outcome,
                Outcome::Rejected(Reject::DivByZero | Reject::DivOverflow)
            ),
            "{op:?}/{sign:?}/{bits} left the residue unspecified for {outcome:?}"
        );
    }
}

#[test]
fn every_operand_pair_at_narrow_widths() {
    for bits in corners::EXHAUSTIVE_WIDTHS {
        let m = mask(bits);
        for sign in Sign::ALL {
            if sign.is_signed() && !corners::signed_width_ok(bits) {
                continue;
            }
            for op in IntOp::ALL {
                for lhs in 0..=m {
                    for rhs in 0..=m {
                        check_point(op, sign, bits, lhs, bits, rhs);
                    }
                }
            }
        }
    }
}

#[test]
fn the_corner_cross_product_at_every_width() {
    for sign in Sign::ALL {
        for &bits in corners::widths_for(sign.is_signed()) {
            let vals = corners::values(bits);
            for op in IntOp::ALL {
                for &lhs in &vals {
                    for &rhs in &vals {
                        check_point(op, sign, bits, lhs, bits, rhs);
                    }
                }
            }
        }
    }
}

#[test]
fn shifts_across_mixed_operand_widths() {
    for sign in Sign::ALL {
        for (bits, rhs_bits) in corners::shift_width_pairs(sign.is_signed()) {
            let vals = corners::values(bits);
            let amounts = corners::shift_amounts(bits, rhs_bits);
            for op in [IntOp::Shl, IntOp::Shr] {
                for &lhs in &vals {
                    for &rhs in &amounts {
                        check_point(op, sign, bits, lhs, rhs_bits, rhs);
                    }
                }
            }
        }
    }
}

#[test]
fn the_accept_boundary_is_exactly_the_width() {
    // Swept rather than spot-checked: for every width, every amount below it is accepted and every
    // amount from it up to the representable maximum is rejected. An off-by-one anywhere in the
    // model or in an evaluator conforming to it shows up here as a single flipped point.
    for sign in Sign::ALL {
        for &bits in corners::widths_for(sign.is_signed()) {
            for op in [IntOp::Shl, IntOp::Shr] {
                for amount in 0..bits {
                    assert!(
                        !eval(op, sign, bits, 1, MAX_BITS, amount as u128).is_rejected(),
                        "{op:?}/{sign:?}/{bits} rejected a legal amount {amount}"
                    );
                }
                for amount in bits..=(bits + 2).min(MAX_BITS) {
                    assert_eq!(
                        eval(op, sign, bits, 1, MAX_BITS, amount as u128),
                        Outcome::Rejected(Reject::ShiftAmount),
                        "{op:?}/{sign:?}/{bits} accepted an illegal amount {amount}"
                    );
                }
            }
        }
    }
}

#[test]
fn odd_widths_are_swept_too() {
    // Not a power of two anywhere. Nothing in Noir produces these, but several evaluators assume
    // the width is a power of two, so the model has to have a defined answer to compare them
    // against when one of them stops assuming it.
    for &bits in &corners::ODD_WIDTHS {
        for sign in Sign::ALL {
            for op in IntOp::ALL {
                for &lhs in &corners::values(bits) {
                    for &rhs in &corners::values(bits) {
                        check_point(op, sign, bits, lhs, bits, rhs);
                    }
                }
            }
        }
    }
}

// LAYER 3: PROPERTY
// ================================================================================================

/// A width and a raw pattern that fits it. Generated together so an out-of-width operand is
/// unrepresentable rather than filtered — filtering is what makes a property test slow and its
/// shrinking useless.
fn width_and_value(max_bits: usize) -> impl Strategy<Value = (usize, u128)> {
    (1usize..=max_bits).prop_flat_map(|bits| {
        let m = mask(bits);
        (Just(bits), (0u128..=m))
    })
}

fn any_op() -> impl Strategy<Value = IntOp> {
    prop::sample::select(IntOp::ALL.to_vec())
}

proptest! {
    /// The interior of the space, where the corner list has no opinion. This is the layer that
    /// would catch a masking or sign-decode bug that is not corner-shaped.
    #[test]
    fn model_invariants_hold_anywhere_unsigned(
        (bits, lhs) in width_and_value(MAX_BITS),
        rhs in any::<u128>(),
        op in any_op(),
    ) {
        check_point(op, Sign::Unsigned, bits, lhs, bits, rhs & mask(bits));
    }

    #[test]
    fn model_invariants_hold_anywhere_signed(
        (bits, lhs) in width_and_value(MAX_SIGNED_BITS),
        rhs in any::<u128>(),
        op in any_op(),
    ) {
        check_point(op, Sign::Signed, bits, lhs, bits, rhs & mask(bits));
    }

    /// Mixed operand widths, which the hand-written pairs cannot enumerate.
    #[test]
    fn shifts_hold_across_independent_widths(
        (bits, lhs) in width_and_value(MAX_BITS),
        (rhs_bits, rhs) in width_and_value(MAX_BITS),
        shl in any::<bool>(),
    ) {
        let op = if shl { IntOp::Shl } else { IntOp::Shr };
        check_point(op, Sign::Unsigned, bits, lhs, rhs_bits, rhs);
    }

    /// Two's complement round-trips, which everything signed rests on.
    #[test]
    fn signed_encoding_round_trips((bits, raw) in width_and_value(MAX_SIGNED_BITS)) {
        let decoded = decode_signed(bits, raw);
        prop_assert!(fits_signed(bits, decoded));
        prop_assert_eq!(encode_signed(bits, decoded), raw);
        prop_assert!(decoded >= signed_min(bits) && decoded <= signed_max(bits));
    }

    /// An accepted signed result is exactly a result that fits, and a rejected one is exactly one
    /// that does not — stated independently of how `eval` decides it.
    #[test]
    fn signed_addition_rejects_exactly_the_unrepresentable(
        (bits, lhs) in width_and_value(MAX_SIGNED_BITS),
        rhs in any::<u128>(),
    ) {
        let rhs = rhs & mask(bits);
        let sum = decode_signed(bits, lhs) + decode_signed(bits, rhs);
        match eval(IntOp::Add, Sign::Signed, bits, lhs, bits, rhs) {
            Outcome::Value(v) => {
                prop_assert!(fits_signed(bits, sum));
                prop_assert_eq!(decode_signed(bits, v), sum);
            }
            Outcome::Rejected(r) => {
                prop_assert!(!fits_signed(bits, sum));
                prop_assert_eq!(r, Reject::Overflow);
            }
        }
    }

    /// A shift is accepted exactly when the amount is below the width, whatever else is true.
    #[test]
    fn shift_acceptance_depends_only_on_the_amount(
        (bits, lhs) in width_and_value(MAX_BITS),
        amount in any::<u128>(),
        shl in any::<bool>(),
    ) {
        let op = if shl { IntOp::Shl } else { IntOp::Shr };
        let rejected = eval(op, Sign::Unsigned, bits, lhs, MAX_BITS, amount).is_rejected();
        prop_assert_eq!(rejected, amount >= bits as u128);
    }

    /// Comparison is a total order consistent with the reading it names.
    #[test]
    fn comparison_matches_its_reading((bits, lhs) in width_and_value(MAX_SIGNED_BITS), rhs in any::<u128>()) {
        let rhs = rhs & mask(bits);
        prop_assert_eq!(cmp(CmpOp::Lt, Sign::Unsigned, bits, lhs, rhs), lhs < rhs);
        prop_assert_eq!(
            cmp(CmpOp::Lt, Sign::Signed, bits, lhs, rhs),
            decode_signed(bits, lhs) < decode_signed(bits, rhs)
        );
        prop_assert_eq!(cmp(CmpOp::Eq, Sign::Unsigned, bits, lhs, rhs), lhs == rhs);
    }
}
