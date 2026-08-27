//! The corner values every conformance sweep runs over.
//!
//! These are deliberately the *known* corners: sign boundaries, `INT_MIN`, `-1`, the width and
//! the width ± 1 as shift amounts. Random operand generation essentially never produces them, which
//! is why this list is the backbone and `proptest` is the layer on top rather than the other way
//! round.

use crate::{MAX_BITS, MAX_SIGNED_BITS, Raw, mask};

/// The widths every sweep covers.
///
/// `1` is `bool`, and a corner in its own right: it is the only width where the sole negative value
/// is `1`, and where `bits - 1` is `0` so every shift amount masks away.
pub const WIDTHS: [usize; 6] = [1, 8, 16, 32, 64, 128];

/// The widths a signed sweep covers — [`WIDTHS`] without the ones no signed operation may touch.
pub const SIGNED_WIDTHS: [usize; 5] = [1, 8, 16, 32, 64];

/// Widths small enough to sweep *every* operand pair at.
///
/// `8` gives 65 536 pairs per operation, which is a second's work for the whole matrix. Anything
/// wider has to fall back to the corner set.
pub const EXHAUSTIVE_WIDTHS: [usize; 3] = [1, 4, 8];

/// Widths that are not powers of two.
///
/// Kept separate because several evaluators quietly assume a power of two.
pub const ODD_WIDTHS: [usize; 3] = [3, 5, 7];

/// Corner raw patterns for a `bits`-wide operand, deduplicated and masked to the width.
///
/// Covers, in both readings: zero and the small values, the unsigned top, the signed boundary pair
/// and its neighbours, `-1` and `-2`, the powers of two that sit at the width's edges and middle,
/// the two alternating patterns, and the two operands the existing corpus tests use for the case
/// where the readings disagree.
#[must_use]
pub fn values(bits: usize) -> Vec<Raw> {
    assert!((1..=MAX_BITS).contains(&bits));
    let m = mask(bits);
    let mut out = vec![0, 1, 2, 3, m, m - 1];

    if bits >= 2 {
        let sign_bit = 1u128 << (bits - 1);
        // `MIN_S`, `MAX_S`, and one step inside each.
        out.extend([sign_bit, sign_bit - 1, sign_bit + 1, sign_bit - 2]);
        // `-1` is `m` (already present) and `-2` is one below it.
        out.push(m - 2);
    }

    for k in [0, 1, bits / 2, bits - 1] {
        if k < bits {
            out.push(1u128 << k);
        }
    }

    // Alternating patterns catch a mask applied at the wrong width, which uniform values cannot.
    out.push(0x5555_5555_5555_5555_5555_5555_5555_5555 & m);
    out.push(0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA & m);

    // `noir_tests/signed_shift` uses 40 and `specialized_shl_wrap` uses 200, and the pair (200,
    // 100) is the one the `BinaryArithOpKind` doc uses to show the readings disagreeing.
    out.extend([40 & m, 100 & m, 200 & m]);

    out.iter_mut().for_each(|v| *v &= m);
    out.sort_unstable();
    out.dedup();
    out
}

/// Shift amounts to try against a `bits`-wide value, as raw patterns at `rhs_bits`.
///
/// The amounts around `bits` are the point: `bits - 1` is the largest legal one, `bits` is the
/// smallest rejected one, and an evaluator that gets the boundary off by one is wrong on exactly
/// those two. The host widths (63, 64, 65, 127, 128) are here because several evaluators reach for
/// a `u64` or `u128` shift internally and inherit *its* masking rather than the operand's.
#[must_use]
pub fn shift_amounts(bits: usize, rhs_bits: usize) -> Vec<Raw> {
    assert!((1..=MAX_BITS).contains(&bits) && (1..=MAX_BITS).contains(&rhs_bits));
    let m = mask(rhs_bits);
    let mut out = vec![0, 1, 63, 64, 65, 127, 128, m];

    for around in [bits, bits / 2] {
        for delta in [0usize, 1, 2] {
            out.push((around + delta) as u128);
            out.push((around.saturating_sub(delta)) as u128);
        }
    }

    // A negative amount, which reads as a huge magnitude and must be rejected for that reason.
    if rhs_bits >= 2 {
        out.push(1u128 << (rhs_bits - 1));
    }

    out.iter_mut().for_each(|v| *v &= m);
    out.sort_unstable();
    out.dedup();
    out
}

/// Every `(bits, rhs_bits)` pair worth sweeping for a shift.
///
/// The equal-width pair is what Noir itself produces (its elaborator unifies a shift's amount with
/// its value) so it is the case that must be right. The mixed pairs are here because the amount's
/// own width is a real degree of freedom in the model, and two evaluators pass one through at
/// runtime: `instrumenter::binary_arith_op` reads each operand at the width its own `Value::Int`
/// carries, and `hlssa_to_r1cs::arith` reads a narrower amount at the value's width and documents
/// why that is safe. The constant folders do not: `lattice::fold_width` declines a mixed pair
/// outright, because `assert_int_arith_widths` would panic on the IR one would have to come from.
#[must_use]
pub fn shift_width_pairs(signed: bool) -> Vec<(usize, usize)> {
    let widths: &[usize] = if signed { &SIGNED_WIDTHS } else { &WIDTHS };
    let mut out = Vec::new();
    for &bits in widths {
        for &rhs_bits in &[bits, 8, 32, 64, MAX_BITS] {
            out.push((bits, rhs_bits));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The widths a sweep should use for `sign`.
#[must_use]
pub fn widths_for(signed: bool) -> &'static [usize] {
    if signed { &SIGNED_WIDTHS } else { &WIDTHS }
}

/// Assert a width is one a signed operation may use, mirroring the model's own bound.
#[must_use]
pub fn signed_width_ok(bits: usize) -> bool {
    (1..=MAX_SIGNED_BITS).contains(&bits)
}
