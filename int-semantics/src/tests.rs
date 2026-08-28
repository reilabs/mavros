//! Tests of the semantics engendered by the model.
//!
//! It encompasses three kinds of test:
//!
//! 1. The **Pins** are hand-written cases taken straight from Noir's own tests and documentation.
//!    These ensure that the model matches Noir's semantics, rather than simply being internally
//!    consistent.
//! 2. The **Exhaustive** tests check every operand pair at `bits ∈ {1, 4, 8}` for every operation
//!    in `IntOp::ALL`. There is no second sweep over a reading, because an operation names its
//!    own: `UAdd` and `SAdd` are two of the sixteen rather than one operation under two signs.
//!    Structural bugs (masking, sign decode, boundary-off-by-one) are generic across widths, so an
//!    exhaustive sweep here will highlight them.
//! 3. The **Property** tests run over the rest of the domain to provide extra coverage where the
//!    exhaustive tier cannot.

use num_bigint::BigUint;
use proptest::prelude::*;

use crate::{
    CmpOp, IntBits, IntOp, MAX_BITS, MAX_SIGNED_BITS, Outcome, Reject, SignedValue, corners, eval,
    int_bits::FIELD_LIMB_BITS, mask, residue,
};

// PROPTEST UTILITIES
// ================================================================================================

/// A `bits`-wide pattern built from a host word.
///
/// The tests below are written as the numbers they mean, not as [`IntBits`]es: a pin that says
/// `200 + 100` overflows at eight bits is about the arithmetic, and burying it under a
/// construction at every operand would hide exactly the thing it pins. So the host façade is here,
/// in the handful of functions below, and nowhere else.
fn pat(bits: usize, value: u128) -> IntBits {
    IntBits::from_u128(bits, value)
}

/// The host word a pattern denotes, the other half of [`pat`].
fn host(value: &IntBits) -> u128 {
    u128::try_from(value).expect("a pattern no wider than MAX_BITS fits a host word")
}

/// Shorthand for the common equal-width call.
fn ev(op: IntOp, bits: usize, lhs: u128, rhs: u128) -> Outcome {
    eval(op, &pat(bits, lhs), &pat(bits, rhs))
}

/// An accepted `bits`-wide answer, for comparing against [`ev`].
fn val(bits: usize, value: u128) -> Outcome {
    Outcome::Value(pat(bits, value))
}

/// [`ev`]'s accepted answer as a host word, or [`None`] where it rejected.
fn ev_value(op: IntOp, bits: usize, lhs: u128, rhs: u128) -> Option<u128> {
    ev(op, bits, lhs, rhs).value().map(|v| host(&v))
}

/// `residue` as a host word.
fn res(op: IntOp, bits: usize, lhs: u128, rhs_bits: usize, rhs: u128) -> Option<u128> {
    residue(op, &pat(bits, lhs), &pat(rhs_bits, rhs)).map(|v| host(&v))
}

/// A small reading written the way the test means it, since a [`SignedValue`] has no literal.
fn sv(v: i64) -> SignedValue {
    SignedValue::from(v)
}

/// The `bits`-wide pattern denoting the reading `v`, as a host word.
fn enc(bits: usize, v: i64) -> u128 {
    host(&IntBits::from_signed(bits, &sv(v)))
}

/// `reduced_shift_amount` on a host word, carried at the widest width the model admits — which is
/// how an amount reaches it from an evaluator that has not narrowed it.
fn reduced(amount: u128, bits: usize) -> u32 {
    pat(MAX_BITS, amount).reduced_shift_amount(bits)
}

/// `IntBits::compare` on host words.
fn compare(op: CmpOp, bits: usize, lhs: u128, rhs: u128) -> bool {
    pat(bits, lhs).compare(op, &pat(bits, rhs))
}

/// `IntBits::from_packed_limbs` as a host word.
fn limb_int(limbs: &[u64], limb_bits: usize, bits: usize) -> u128 {
    host(&IntBits::from_packed_limbs(limbs, limb_bits, bits))
}

/// `IntBits::from_field_limbs` as a host word.
fn field_int(limbs: &[u64], bits: usize) -> u128 {
    host(&IntBits::from_field_limbs(limbs, bits))
}

// LAYER 1: PINS AGAINST NOIR
// ================================================================================================

#[test]
fn arithmetic_overflow_is_rejected_not_wrapped() {
    // `eval_constant_binary_op` computes the result and then rejects it when `result.ilog2() >=
    // bit_size`. This is the rule mavros currently breaks for unguarded pure operations: every
    // backend wraps to 44 and the program runs clean.
    assert_eq!(
        ev(IntOp::UAdd, 8, 200, 100),
        Outcome::Rejected(Reject::Overflow)
    );
    // ... and the residue records what those backends compute, so a conformance test can hold them
    // to it without pretending it is the right answer.
    assert_eq!(res(IntOp::UAdd, 8, 200, 8, 100), Some(44));

    // Unsigned subtraction below zero is an overflow, not a wrap to a large number.
    assert_eq!(
        ev(IntOp::USub, 8, 1, 2),
        Outcome::Rejected(Reject::Overflow)
    );
    assert_eq!(res(IntOp::USub, 8, 1, 8, 2), Some(255));

    assert_eq!(
        ev(IntOp::UMul, 8, 16, 16),
        Outcome::Rejected(Reject::Overflow)
    );

    // Signed overflow is the same story at the signed boundary.
    let max = host(&IntBits::from_signed(8, &IntBits::signed_max(8)));
    assert_eq!(
        ev(IntOp::SAdd, 8, max, 1),
        Outcome::Rejected(Reject::Overflow)
    );
    assert_eq!(
        res(IntOp::SAdd, 8, max, 8, 1),
        Some(host(&IntBits::from_signed(8, &IntBits::signed_min(8)))),
        "127i8 + 1 wraps to -128 in every backend"
    );

    // In range, so accepted, and the two readings differ on the same patterns.
    assert_eq!(ev(IntOp::UAdd, 8, 200, 55), val(8, 255));
    assert_eq!(ev(IntOp::SAdd, 8, 200, 55), val(8, 255));
}

#[test]
fn division_rejects_a_zero_divisor_and_the_signed_overflow() {
    for op in [IntOp::UDiv, IntOp::SDiv, IntOp::URem, IntOp::SRem] {
        assert_eq!(ev(op, 8, 7, 0), Outcome::Rejected(Reject::DivByZero));
        assert_eq!(res(op, 8, 7, 8, 0), None, "no agreed answer exists");
    }

    // `INT_MIN / -1` has a quotient one past the top of the type. `expand_signed_math` emits an
    // explicit constrain for exactly this pair.
    let min = host(&IntBits::from_signed(8, &IntBits::signed_min(8)));
    let minus_one = enc(8, -1);
    assert_eq!(
        ev(IntOp::SDiv, 8, min, minus_one),
        Outcome::Rejected(Reject::DivOverflow)
    );
    // `INT_MIN % -1` would produce 0, which is representable -- but Noir defines the remainder in
    // terms of the same quotient, so it rejects that too.
    assert_eq!(
        ev(IntOp::SRem, 8, min, minus_one),
        Outcome::Rejected(Reject::DivOverflow)
    );
    // The unsigned reading of the same patterns is 128 / 255, which is perfectly fine.
    assert_eq!(ev(IntOp::UDiv, 8, min, minus_one), val(8, 0));
}

#[test]
fn signed_division_truncates_toward_zero() {
    // `noir_tests/signed_divmod` pins this, and it is what separates `/` from an arithmetic `>>`:
    // `-7 / 2` is -3 while `-7 >> 1` is -4.
    let neg7 = enc(8, -7);
    assert_eq!(ev_value(IntOp::SDiv, 8, neg7, 2), Some(enc(8, -3)));
    assert_eq!(ev_value(IntOp::SRem, 8, neg7, 2), Some(enc(8, -1)));
    assert_eq!(ev_value(IntOp::SShr, 8, neg7, 1), Some(enc(8, -4)));

    // The remainder follows the dividend, not the divisor.
    let neg2 = enc(8, -2);
    assert_eq!(ev_value(IntOp::SDiv, 8, 7, neg2), Some(enc(8, -3)));
    assert_eq!(ev_value(IntOp::SRem, 8, 7, neg2), Some(1));
}

#[test]
fn a_left_shift_wraps_and_only_the_amount_can_reject() {
    // `noir_tests/specialized_shl_wrap` exists to pin this: 200 << 1 is 400, which keeps only its
    // low eight bits, so the answer is 144 and *not* a rejection. Losing bits off the top is the
    // specified behaviour of `<<`, unlike `*`, where it is an error.
    assert_eq!(ev(IntOp::Shl, 8, 200, 1), val(8, 144));
    assert_eq!(
        ev(IntOp::UMul, 8, 200, 2),
        Outcome::Rejected(Reject::Overflow)
    );

    // A signed `<<` wraps across the sign boundary: 64i8 << 1 is -128.
    assert_eq!(
        ev_value(IntOp::Shl, 8, 64, 1),
        Some(host(&IntBits::from_signed(8, &IntBits::signed_min(8))))
    );
    // ... which is why the model shifts the raw pattern rather than asking whether `64 * 2` fits.
    assert_eq!(ev_value(IntOp::Shl, 8, 1, 7), Some(0x80));
}

#[test]
fn a_shift_amount_at_or_above_the_width_is_rejected() {
    // `remove_bit_shifts`' module doc: "In all cases, if the shift amount is equal to or exceeds
    // the operand's number of bits, the result will be a constrain failure". The boundary is the
    // whole content of the rule, so both sides of it are pinned.
    for op in [IntOp::Shl, IntOp::UShr, IntOp::SShr] {
        assert!(
            ev_value(op, 8, 1, 7).is_some(),
            "7 is the largest legal amount"
        );
        assert_eq!(ev(op, 8, 1, 8), Outcome::Rejected(Reject::ShiftAmount));
        assert_eq!(ev(op, 8, 1, 100), Outcome::Rejected(Reject::ShiftAmount));
    }

    // Noir's own `shl_overflow_u64` / `shr_overflow_u64` are a u64 shifted by 100.
    for op in [IntOp::Shl, IntOp::UShr] {
        assert_eq!(ev(op, 64, 1, 100), Outcome::Rejected(Reject::ShiftAmount));
    }

    // A *negative* amount is the same rejection and not a separate rule:
    // `enforce_bitshift_rhs_lt_bit_size` casts the amount to unsigned first, so -1 reads as a huge
    // magnitude. `shl_signed_regression_9592` is this case.
    let minus_one = enc(32, -1);
    assert_eq!(
        ev(IntOp::Shl, 32, 1, minus_one),
        Outcome::Rejected(Reject::ShiftAmount)
    );
}

#[test]
fn a_right_shift_fills_by_its_reading() {
    // Unsigned zero-fills, signed sign-fills and saturates at -1 rather than reaching zero.
    assert_eq!(ev(IntOp::UShr, 8, 0xF0, 2), val(8, 0x3C));
    assert_eq!(ev(IntOp::SShr, 8, 0xF0, 2), val(8, 0xFC));
    assert_eq!(ev_value(IntOp::SShr, 8, 0xF0, 7), Some(enc(8, -1)));
    assert_eq!(ev(IntOp::SShr, 8, 0, 7), val(8, 0));
}

#[test]
fn bitwise_operations_have_no_reading_and_never_reject() {
    // There is no sign to sweep: these three have no signed form to disagree with, which is this
    // test's claim made structural rather than checked.
    assert_eq!(ev(IntOp::And, 8, 0xF0, 0x3C), val(8, 0x30));
    assert_eq!(ev(IntOp::Or, 8, 0xF0, 0x3C), val(8, 0xFC));
    assert_eq!(ev(IntOp::Xor, 8, 0xF0, 0x3C), val(8, 0xCC));
    for op in [IntOp::And, IntOp::Or, IntOp::Xor] {
        assert_eq!(op.sign(), None, "{op:?} must name no reading");
    }
}

#[test]
fn comparison_reads_the_patterns_as_the_operation_says() {
    // 0xFB is 251 unsigned and -5 signed, so the two readings order it against 2 differently.
    assert!(compare(CmpOp::SLt, 8, 0xFB, 2));
    assert!(!compare(CmpOp::ULt, 8, 0xFB, 2));
    // Equality needs no reading at all, which is why there is one `Eq` rather than a pair.
    assert!(compare(CmpOp::Eq, 8, 0xFB, 0xFB));
    assert!(!compare(CmpOp::Eq, 8, 0xFB, 2));
    assert_eq!(CmpOp::Eq.sign(), None);
}

#[test]
fn the_field_cast_reads_both_low_limbs() {
    // The bug this exists to prevent: reading only `limbs[0]` answers a different number for every
    // `Field as u128` above 2^64, which is exactly the range `u128` was added for.
    let limbs = [7u64, 1u64, 0, 0];
    assert_eq!(field_int(&limbs, 128), (1u128 << 64) | 7);
    assert_eq!(field_int(&limbs, 64), 7, "narrowing still truncates");
    assert_eq!(field_int(&limbs, 8), 7);
}

#[test]
fn the_field_cast_is_indifferent_to_the_limb_count() {
    // bn254 has four limbs and a smaller field has fewer, so the count is the field's business and
    // not this crate's. Every one of these describes the same element.
    assert_eq!(field_int(&[7u64], 128), 7);
    assert_eq!(field_int(&[7u64, 0, 0], 128), 7);
    assert_eq!(field_int(&[7u64, 0, 0, 0, 0, 0], 128), 7);

    // Limbs past the second hold only bits `mask` discards, so a longer slice cannot change the
    // answer -- the guard against a future width picking them up silently.
    assert_eq!(
        field_int(&[7u64, 1, u64::MAX, u64::MAX], 128),
        (1u128 << 64) | 7
    );
    assert_eq!(field_int(&[], 128), 0, "no limbs is the zero element");
}

#[test]
fn the_fit_test_covers_exactly_the_bits_the_read_keeps() {
    // The companion of the truncating read above: whatever `IntBits::field_limbs_fit` accepts,
    // `IntBits::from_field_limbs` must have lost nothing from.
    assert!(IntBits::field_limbs_fit(&[7u64, 1, 0, 0], 128));
    assert!(!IntBits::field_limbs_fit(&[7u64, 1, 1, 0], 128));
    assert!(!IntBits::field_limbs_fit(&[0u64, 0, 0, 1], 128));
    assert!(
        IntBits::field_limbs_fit(&[], 128),
        "no limbs is the zero element"
    );

    // A width that does not end on a limb boundary is the case the obvious spelling gets wrong:
    // `70 / 64` is `1`, so "every limb from index 1 up is zero" would reject an element whose
    // seventieth bit is the highest one set, which fits perfectly well.
    assert!(IntBits::field_limbs_fit(&[u64::MAX, 0b11_1111, 0, 0], 70));
    assert!(!IntBits::field_limbs_fit(&[u64::MAX, 0b111_1111, 0, 0], 70));

    // And the boundary width itself, where the limb above must be empty outright.
    assert!(IntBits::field_limbs_fit(&[u64::MAX, 0, 0, 0], 64));
    assert!(!IntBits::field_limbs_fit(&[u64::MAX, 1, 0, 0], 64));

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
            if !IntBits::field_limbs_fit(&limbs, bits) {
                continue;
            }
            assert_eq!(
                field_int(&limbs, bits),
                field_int(&limbs, MAX_BITS),
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
        limb_int(&canonical, FIELD_LIMB_BITS, 128),
        field_int(&canonical, 128)
    );

    // The witness decompositions in `witness_bitwise` use a limb width derived from the field size,
    // so the same little-endian vector says something different at a different width.
    assert_eq!(limb_int(&[7u64, 1, 0, 0], 32, 128), (1u128 << 32) | 7);
    assert_eq!(limb_int(&[1u64, 0, 1], 1, 128), 0b101);

    // A narrow limb packed into a wider container carries junk above its width, which is the
    // caller's normal state rather than an error.
    assert_eq!(limb_int(&[u64::MAX], 32, 128), u128::from(u32::MAX));

    // A limb at least as wide as the result needs no recombination at all.
    assert_eq!(limb_int(&[7u64], MAX_BITS, 128), 7);
}

#[test]
#[should_panic(expected = "a limb must be at least one bit wide")]
fn a_zero_width_limb_describes_no_representation() {
    let _ = limb_int(&[7u64], 0, 128);
}

// SIGN EXTENSION
// ================================================================================================
//
// These are the tests that sweep the width extended *from*, which is what distinguishes
// `sign_extend` from a widening cast: an implementation filling from a fixed width passes every
// other test in this file.

/// `IntBits::sign_extend` on host words.
fn sext(raw: u128, from: usize, to: usize) -> u128 {
    host(&pat(from, raw).sign_extend(to))
}

#[test]
fn sign_extend_agrees_with_decode_encode() {
    // The authoritative definition of sign extension is "same integer, wider encoding".
    // Check the bit-twiddling against it exhaustively for every 4-bit and 8-bit value.
    for from in [4usize, 8] {
        for raw in 0..(1u128 << from) {
            for to in [from, from + 1, 16, 32, 64] {
                assert_eq!(
                    sext(raw, from, to),
                    host(&IntBits::from_signed(to, &pat(from, raw).to_signed())),
                    "sign_extend({raw} at {from} bits, {to})"
                );
            }
        }
    }
}

#[test]
fn sign_extend_is_identity_at_equal_width() {
    for from in [1usize, 7, 8, 32] {
        for raw in [0u128, 1, mask(from), mask(from) >> 1] {
            assert_eq!(sext(raw, from, from), raw & mask(from));
        }
    }
}

#[test]
fn sign_extend_masks_its_input() {
    // Bits above `from` are not part of the value and must not survive into the result. The
    // masking happens one step earlier: an eight-bit pattern cannot carry a ninth bit, so building
    // one from a wider host word is where the discarding is done.
    assert_eq!(sext(0xFF00 | 0x01, 8, 16), 0x0001);
    assert_eq!(sext(0xFF00 | 0x80, 8, 16), 0xFF80);
}

#[test]
fn sign_extend_handles_full_width() {
    // The full width must not overflow anything the fill is built from.
    assert_eq!(sext(1, 1, 128), u128::MAX);
    assert_eq!(sext(0, 1, 128), 0);
}

// THE VOCABULARY
// ================================================================================================
//
// The reading lives on the operation rather than beside it. What these check is the *shape* of the
// vocabulary: that the operations which behave differently are two, and the ones that behave
// identically are one.

#[test]
fn every_operation_that_reads_its_operands_comes_in_a_pair() {
    // Stated as a partition rather than a list, so that adding a variant without a partner, or a
    // partner without a difference, fails here rather than being noticed later.
    let mut paired: Vec<&str> = Vec::new();
    let mut single: Vec<&str> = Vec::new();
    for op in IntOp::ALL {
        let name = op.name();
        if op.sign().is_some() {
            if !paired.contains(&name) {
                paired.push(name);
            }
        } else {
            single.push(name);
        }
    }
    assert_eq!(paired, ["add", "sub", "mul", "div", "rem", "shr"]);
    assert_eq!(single, ["and", "or", "xor", "shl"]);

    // Each paired name really is two operations, and each single name really is one.
    for name in paired {
        let members: Vec<IntOp> = IntOp::ALL
            .into_iter()
            .filter(|o| o.name() == name)
            .collect();
        assert_eq!(members.len(), 2, "{name} is not a pair");
        assert_ne!(
            members[0].sign(),
            members[1].sign(),
            "{name} reads one way twice"
        );
    }
}

#[test]
fn the_operations_with_no_reading_are_the_ones_that_answer_the_same_either_way() {
    // The claim behind `And`/`Or`/`Xor`/`Shl` having no signed form, checked rather than asserted
    // in prose: for every corner pair, the answer read as *signed* is the one signed arithmetic
    // would have produced, and the answer read as *unsigned* is the one unsigned arithmetic would
    // have produced. Both are derived here through `SignedValue`/`BigUint` rather than through
    // `IntBits`'s own limb code, so this compares two routes rather than restating one.
    //
    // A signed `<<` differs from an unsigned one only in a rejecting constraint on a negative
    // *amount*, which is guard IR and not something `eval` computes.
    for bits in [1usize, 8, 64] {
        for &lhs in &corners::values(bits) {
            for &rhs in &corners::values(bits) {
                let (l, r) = (pat(bits, lhs), pat(bits, rhs));

                // The bitwise three, each derived twice: once over `BigInt`, whose operators read
                // a negative as two's complement extended indefinitely, and once over `BigUint`,
                // which is the plain magnitude. One answer has to satisfy both, and that is what
                // "the same either way" means.
                let (ls, rs) = (l.to_signed(), r.to_signed());
                let (lu, ru) = (BigUint::from(&l), BigUint::from(&r));
                for (op, signed_want, unsigned_want) in [
                    (IntOp::And, &ls & &rs, &lu & &ru),
                    (IntOp::Or, &ls | &rs, &lu | &ru),
                    (IntOp::Xor, &ls ^ &rs, &lu ^ &ru),
                ] {
                    let Outcome::Value(got) = ev(op, bits, lhs, rhs) else {
                        panic!("{op:?} cannot reject");
                    };
                    assert_eq!(
                        got.to_signed(),
                        signed_want,
                        "{op:?} at {bits} on {lhs:#x} {rhs:#x} read as signed"
                    );
                    assert_eq!(
                        BigUint::from(&got),
                        unsigned_want,
                        "{op:?} at {bits} on {lhs:#x} {rhs:#x} read as unsigned"
                    );
                }

                // `Shl` is the one that has to be argued. Under the unsigned reading it is
                // `lhs * 2^amount` truncated to the width; under the signed reading it is
                // `lhs_signed * 2^amount` encoded back into the width. Both are the same bits,
                // which is why there is one variant and not two.
                let shifted = ev(IntOp::Shl, bits, lhs, rhs);
                if rhs < bits as u128 {
                    let amount = rhs as usize;
                    let unsigned = IntBits::from_biguint(bits, &(BigUint::from(&l) << amount));
                    let signed = IntBits::from_signed(bits, &(l.to_signed() << amount));
                    assert_eq!(unsigned, signed, "the two readings of `<<` are one map");
                    assert_eq!(
                        shifted,
                        Outcome::Value(unsigned),
                        "shl at {bits} on {lhs:#x} {rhs:#x}"
                    );
                } else {
                    assert_eq!(shifted, Outcome::Rejected(Reject::ShiftAmount));
                }
            }
        }
    }
}

#[test]
fn the_shift_pair_is_the_one_that_differs() {
    // `>>` is where the reading changes the answer, which is why it splits where `<<` does not.
    assert_eq!(ev_value(IntOp::UShr, 8, 0xF0, 4), Some(0x0F));
    assert_eq!(ev_value(IntOp::SShr, 8, 0xF0, 4), Some(0xFF));

    // Stated as the exact rule rather than the one pair above, so that the split is pinned by
    // where the two forms part company: a non-zero amount fills the vacated top bits, and the two
    // fill it differently exactly when there is a sign bit set to fill it with.
    for bits in [1usize, 8, 32, 64] {
        for &lhs in &corners::values(bits) {
            let negative = pat(bits, lhs).to_signed() < sv(0);
            for amount in 0..bits as u128 {
                let logical = ev_value(IntOp::UShr, bits, lhs, amount);
                let arithmetic = ev_value(IntOp::SShr, bits, lhs, amount);
                assert_eq!(
                    logical != arithmetic,
                    negative && amount != 0,
                    "{lhs:#x} >> {amount} at {bits} bits: {logical:?} vs {arithmetic:?}"
                );
            }
        }
    }
}

// THE WIDTHS THE OPERANDS ARRIVE CARRYING
// ================================================================================================
//
// `check_widths` had no coverage at all while the widths were parameters, because a caller that
// spelled them wrong was writing a bug in its own call rather than building a value. They come off
// the operands now, so what these pin is that the model still refuses a pair no operation admits
// -- and, in the shift's case, still admits the one pair that is legitimately mixed.

#[test]
#[should_panic(expected = "only a shift reads its right operand at a width of its own")]
fn a_non_shift_refuses_two_widths() {
    let _ = eval(IntOp::UAdd, &pat(8, 1), &pat(16, 1));
}

#[test]
fn a_shift_takes_its_amount_at_whatever_width_it_arrives() {
    // Noir's elaborator unifies a shift's amount with its value, but the IR does not require it and
    // two evaluators pass a narrower amount through at runtime -- see `corners::shift_width_pairs`.
    assert_eq!(eval(IntOp::Shl, &pat(8, 1), &pat(32, 3)), val(8, 8));
    assert_eq!(
        eval(IntOp::Shl, &pat(8, 1), &pat(MAX_BITS, 8)),
        Outcome::Rejected(Reject::ShiftAmount),
        "the bound is the value's width, not the amount's"
    );
}

#[test]
#[should_panic(expected = "an integer pattern is at least one bit wide")]
fn a_zero_width_operand_cannot_even_be_built() {
    // `check_widths` states `1..=MAX_BITS`, but the lower half of that range is not its to
    // enforce: a width of zero is refused a whole layer earlier, when the operand is constructed.
    // The upper half is live, since `MAX_BITS` is the model's cap rather than the type's.
    let _ = eval(IntOp::UAdd, &pat(0, 0), &pat(0, 0));
}

#[test]
#[should_panic(expected = "operand width 200 is outside")]
fn an_operand_wider_than_the_model_admits_is_refused() {
    let _ = eval(IntOp::UAdd, &pat(200, 1), &pat(200, 1));
}

#[test]
#[should_panic(expected = "signed operation on a 128-bit value")]
fn a_signed_reading_stops_where_the_lowerings_do() {
    let _ = eval(IntOp::SAdd, &pat(128, 1), &pat(128, 1));
}

#[test]
#[should_panic(expected = "only a shift reads its right operand at a width of its own")]
fn a_comparison_refuses_two_widths() {
    // `compare` reads a width off each operand rather than being handed one for both, which is
    // what makes this a check rather than a convention -- and it is a live question, because
    // HLSSA's `Cmp` typing rule does not require equal widths and the instrumenter's own
    // comparison decides one.
    let _ = pat(8, 1).compare(CmpOp::ULt, &pat(16, 1));
}

#[test]
fn bit_level_helpers_agree_with_their_definitions() {
    // Each operand states the width it came from, which is what these helpers read: the width
    // `sign_extend` extends *from* is the operand's own rather than a second opinion beside it.
    assert_eq!(host(&pat(9, 0x1FF).cast(8)), 0xFF);
    assert_eq!(
        host(&pat(8, 0xFF).cast(32)),
        0xFF,
        "widening a raw pattern zero-extends"
    );
    assert_eq!(
        host(&pat(8, 0xFF).sign_extend(32)),
        0xFFFF_FFFF,
        "widening a signed one does not"
    );
    assert_eq!(host(&pat(8, 0x7F).sign_extend(32)), 0x7F);
    assert_eq!(host(&pat(16, 0xABCD).bit_range(8, 8)), 0xAB);
    assert_eq!(host(&pat(16, 0xABCD).bit_range(0, 8)), 0xCD);
    assert_eq!(host(&pat(8, 0xF0).complement()), 0x0F);
    assert_eq!(pat(8, 0xFF).to_signed(), sv(-1));
    assert_eq!(
        pat(1, 1).to_signed(),
        sv(-1),
        "the only negative a one-bit pattern holds"
    );
    assert_eq!(enc(8, -1), 0xFF);
}

#[test]
fn the_host_word_mask_saturates_at_the_host_width_not_the_models_cap() {
    // `mask` is the last host word in the model, and its saturating arm exists because
    // `1u128 << bits` is a debug panic and a release build that silently masks the shift amount.
    // The bound has to be the **host's** 128 rather than `MAX_BITS`, which the plan moves.
    assert_eq!(mask(0), 0);
    assert_eq!(mask(1), 1);
    assert_eq!(mask(64), u128::from(u64::MAX));
    assert_eq!(mask(127), u128::MAX >> 1);
    assert_eq!(mask(128), u128::MAX);

    // The tripwire. Today `MAX_BITS` is 128 and these are the same assertion as the one above; the
    // moment the cap moves they stop being, and a bound written against it would take the `<<` arm
    // for a width no `u128` can express.
    assert_eq!(mask(129), u128::MAX);
    assert_eq!(mask(1000), u128::MAX);
    assert_eq!(mask(MAX_BITS + 1), u128::MAX);
}

#[test]
fn a_reading_outside_the_width_encodes_by_wrapping() {
    // `IntBits::from_signed` is total, which it has to be for an unbounded `SignedValue`: an
    // intermediate result legitimately leaves the width, and the answer is the low `bits` of its
    // two's complement. That is a claim about `BigInt`'s bitwise operators reading a negative as
    // two's complement extended indefinitely, so it is pinned rather than assumed.
    assert_eq!(enc(8, 256), 0, "one whole turn is back where it started");
    assert_eq!(enc(8, 257), 1);
    assert_eq!(enc(8, -129), 0x7F, "one step past the bottom of the type");
    assert_eq!(enc(8, -1000), u128::from(-1000i32 as u32 & 0xFF));
    assert_eq!(enc(1, -1), 1, "the whole of a one-bit type");
}

#[test]
fn the_signed_boundaries_are_total_at_every_width() {
    // The two boundaries are pure powers of two and nothing caps them at `MAX_SIGNED_BITS`, so
    // they are the first part of the model to reach the full width — and the part a host type
    // could not have carried. `1i128 << 127` is not representable, which is what made the obvious
    // spelling of the signed reading panic at width 127.
    for bits in 1..=MAX_BITS {
        let (min, max) = (IntBits::signed_min(bits), IntBits::signed_max(bits));
        assert_eq!(
            max.clone() + SignedValue::from(1u8),
            -min.clone(),
            "the top of a {bits}-bit type is one below the negated bottom"
        );
        assert_eq!(
            max - min,
            (SignedValue::from(1u8) << bits) - SignedValue::from(1u8),
            "a {bits}-bit type spans 2^{bits} values"
        );
    }
    assert_eq!(IntBits::signed_max(128), SignedValue::from(i128::MAX));
    assert_eq!(IntBits::signed_min(128), SignedValue::from(i128::MIN));
}

#[test]
fn the_shift_backstop_reduces_to_the_operand_width_not_the_host() {
    // The clause that catches an evaluator inheriting a host width: `u128::wrapping_shl` masks to
    // 127, so an evaluator built on it answers `1u8 << 8 == 0` where the VM and LLVM, which reduce
    // by `bits`, answer `1`. `hlssa_to_r1cs::arith` avoids that by calling `residue` rather than
    // shifting a host word itself, so this is the statement it relies on.
    assert_eq!(reduced(8, 8), 0);
    assert_eq!(reduced(100, 64), 36);
    assert_eq!(
        res(IntOp::Shl, 8, 1, 8, 8),
        Some(1),
        "reducing by the operand width, not by the host's"
    );

    // At a non-power-of-two width a mask is a submask rather than a modulo: `9 & 6` is `0`, which
    // is not even the amount `9` reduces to. The modulo answers `9 % 7 == 2`, and an in-range
    // amount is left exactly alone.
    assert_eq!(reduced(9, 7), 2);
    assert_eq!(reduced(1, 7), 1, "an in-range amount is untouched");
    assert!(reduced(u128::MAX, 7) < 7);

    // Every width, not only the powers of two: an amount already below the width survives.
    for bits in 1..=MAX_BITS {
        for amount in 0..bits {
            assert_eq!(
                reduced(amount as u128, bits),
                amount as u32,
                "an in-range amount must be left alone at {bits} bits"
            );
        }
    }
}

// LAYER 2: EXHAUSTIVE
// ================================================================================================

/// The invariant that makes the two-function split safe: a residue never contradicts an accepted
/// evaluation, it only extends it into the rejected region.
fn check_point(op: IntOp, bits: usize, lhs: u128, rhs_bits: usize, rhs: u128) {
    let outcome = eval(op, &pat(bits, lhs), &pat(rhs_bits, rhs));
    let residue = res(op, bits, lhs, rhs_bits, rhs);

    if let Outcome::Value(v) = &outcome {
        // "Inside its width" is a statement about the answer's _declared_ width rather than about
        // its magnitude: a pattern cannot carry a bit above the width it says it has, so what is
        // left to check is that the model answers at the width it was asked at.
        assert_eq!(
            v.bits(),
            bits,
            "{op:?}/{bits} answered at width {} instead, from {lhs:#x} {rhs:#x}",
            v.bits()
        );
        assert_eq!(
            residue,
            Some(host(v)),
            "{op:?}/{bits} residue contradicts its own accepted value"
        );
        if op.is_signed() {
            assert!(
                IntBits::fits_signed(bits, &v.to_signed()),
                "{op:?}/{bits} produced a pattern that does not decode into its own width"
            );
        }
    }

    if let Some(r) = residue {
        assert!(
            r <= mask(bits),
            "{op:?}/{bits} residue {r:#x} escaped its width"
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
            "{op:?}/{bits} left the residue unspecified for {outcome:?}"
        );
    }
}

#[test]
fn every_operand_pair_at_narrow_widths() {
    for bits in corners::EXHAUSTIVE_WIDTHS {
        let m = mask(bits);
        for op in IntOp::ALL {
            if op.is_signed() && !corners::signed_width_ok(bits) {
                continue;
            }
            for lhs in 0..=m {
                for rhs in 0..=m {
                    check_point(op, bits, lhs, bits, rhs);
                }
            }
        }
    }
}

#[test]
fn the_corner_cross_product_at_every_width() {
    for op in IntOp::ALL {
        for &bits in corners::widths_for(op.is_signed()) {
            let vals = corners::values(bits);
            for &lhs in &vals {
                for &rhs in &vals {
                    check_point(op, bits, lhs, bits, rhs);
                }
            }
        }
    }
}

#[test]
fn shifts_across_mixed_operand_widths() {
    for op in [IntOp::Shl, IntOp::UShr, IntOp::SShr] {
        for (bits, rhs_bits) in corners::shift_width_pairs(op.is_signed()) {
            let vals = corners::values(bits);
            let amounts = corners::shift_amounts(bits, rhs_bits);
            for &lhs in &vals {
                for &rhs in &amounts {
                    check_point(op, bits, lhs, rhs_bits, rhs);
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
    for op in [IntOp::Shl, IntOp::UShr, IntOp::SShr] {
        for &bits in corners::widths_for(op.is_signed()) {
            for amount in 0..bits {
                assert!(
                    !eval(op, &pat(bits, 1), &pat(MAX_BITS, amount as u128)).is_rejected(),
                    "{op:?}/{bits} rejected a legal amount {amount}"
                );
            }
            for amount in bits..=(bits + 2).min(MAX_BITS) {
                assert_eq!(
                    eval(op, &pat(bits, 1), &pat(MAX_BITS, amount as u128)),
                    Outcome::Rejected(Reject::ShiftAmount),
                    "{op:?}/{bits} accepted an illegal amount {amount}"
                );
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
        for op in IntOp::ALL {
            for &lhs in &corners::values(bits) {
                for &rhs in &corners::values(bits) {
                    check_point(op, bits, lhs, bits, rhs);
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

/// Any operation whose reading is not signed, so the sweep may use the full width range.
///
/// The reading-free operations belong here rather than being dropped: `And` at 128 bits is
/// perfectly legal and is the case the unsigned sweep is for.
fn any_unsigned_op() -> impl Strategy<Value = IntOp> {
    prop::sample::select(
        IntOp::ALL
            .into_iter()
            .filter(|op| !op.is_signed())
            .collect::<Vec<_>>(),
    )
}

/// Any operation that reads its operands as two's complement.
fn any_signed_op() -> impl Strategy<Value = IntOp> {
    prop::sample::select(
        IntOp::ALL
            .into_iter()
            .filter(|op| op.is_signed())
            .collect::<Vec<_>>(),
    )
}

proptest! {
    /// The interior of the space, where the corner list has no opinion. This is the layer that
    /// would catch a masking or sign-decode bug that is not corner-shaped.
    #[test]
    fn model_invariants_hold_anywhere_unsigned(
        (bits, lhs) in width_and_value(MAX_BITS),
        rhs in any::<u128>(),
        op in any_unsigned_op(),
    ) {
        check_point(op, bits, lhs, bits, rhs & mask(bits));
    }

    #[test]
    fn model_invariants_hold_anywhere_signed(
        (bits, lhs) in width_and_value(MAX_SIGNED_BITS),
        rhs in any::<u128>(),
        op in any_signed_op(),
    ) {
        check_point(op, bits, lhs, bits, rhs & mask(bits));
    }

    /// Mixed operand widths, which the hand-written pairs cannot enumerate.
    #[test]
    fn shifts_hold_across_independent_widths(
        (bits, lhs) in width_and_value(MAX_BITS),
        (rhs_bits, rhs) in width_and_value(MAX_BITS),
        shl in any::<bool>(),
    ) {
        let op = if shl { IntOp::Shl } else { IntOp::UShr };
        check_point(op, bits, lhs, rhs_bits, rhs);
    }

    /// Two's complement round-trips, which everything signed rests on.
    #[test]
    fn signed_encoding_round_trips((bits, raw) in width_and_value(MAX_SIGNED_BITS)) {
        let decoded = pat(bits, raw).to_signed();
        prop_assert!(IntBits::fits_signed(bits, &decoded));
        prop_assert_eq!(host(&IntBits::from_signed(bits, &decoded)), raw);
        prop_assert!(decoded >= IntBits::signed_min(bits) && decoded <= IntBits::signed_max(bits));
    }

    /// An accepted signed result is exactly a result that fits, and a rejected one is exactly one
    /// that does not — stated independently of how `eval` decides it.
    #[test]
    fn signed_addition_rejects_exactly_the_unrepresentable(
        (bits, lhs) in width_and_value(MAX_SIGNED_BITS),
        rhs in any::<u128>(),
    ) {
        let rhs = rhs & mask(bits);
        let sum = pat(bits, lhs).to_signed() + pat(bits, rhs).to_signed();
        match ev(IntOp::SAdd, bits, lhs, rhs) {
            Outcome::Value(v) => {
                prop_assert!(IntBits::fits_signed(bits, &sum));
                prop_assert_eq!(v.to_signed(), sum);
            }
            Outcome::Rejected(r) => {
                prop_assert!(!IntBits::fits_signed(bits, &sum));
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
        let op = if shl { IntOp::Shl } else { IntOp::UShr };
        let rejected =
            eval(op, &pat(bits, lhs), &pat(MAX_BITS, amount))
                .is_rejected();
        prop_assert_eq!(rejected, amount >= bits as u128);
    }

    /// Comparison is a total order consistent with the reading it names.
    #[test]
    fn comparison_matches_its_reading((bits, lhs) in width_and_value(MAX_SIGNED_BITS), rhs in any::<u128>()) {
        let rhs = rhs & mask(bits);
        prop_assert_eq!(compare(CmpOp::ULt, bits, lhs, rhs), lhs < rhs);
        prop_assert_eq!(
            compare(CmpOp::SLt, bits, lhs, rhs),
            pat(bits, lhs).to_signed() < pat(bits, rhs).to_signed()
        );
        prop_assert_eq!(compare(CmpOp::Eq, bits, lhs, rhs), lhs == rhs);
    }
}
