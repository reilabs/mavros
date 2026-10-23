//! The corner values every conformance sweep runs over: sign boundaries, `INT_MIN`, `-1`, and the
//! width and the width ± 1 as shift amounts. Random generation almost never produces them, so this
//! list is the backbone and `proptest` a layer on top.

use std::sync::LazyLock;

use num_bigint::BigUint;

use crate::{
    IntBits, IntOp,
    int_bits::{HOST_LIMB_BITS, HOST_WORD_BITS},
    mask,
};

/// The widest pattern the generators below can express.
///
/// They return host words (`Vec<u128>`), so this is a limit of the return type and deliberately
/// **not** [`MAX_BITS`](crate::MAX_BITS): `values(16384)` would sweep a 128-bit corner set at a
/// 16384-bit width and pass while checking almost nothing.
///
/// Wide corners come from separate generators returning [`IntBits`].
const HOST_CORNER_BITS: usize = HOST_WORD_BITS;

/// The widths every sweep covers.
///
/// `1` is `bool`, and a corner in its own right: it is the only width where the sole negative value
/// is `1`, and where `bits - 1` is `0` so every shift amount masks away.
pub const WIDTHS: [usize; 6] = [1, 8, 16, 32, 64, 128];

/// Widths small enough to sweep _every_ operand pair at.
///
/// `8` gives 65 536 pairs per operation, which is a second's work for the whole matrix. Anything
/// wider has to fall back to the corner set.
pub const EXHAUSTIVE_WIDTHS: [usize; 3] = [1, 4, 8];

/// Widths that are not powers of two. A mask by `bits - 1` passes at every width in [`WIDTHS`] and
/// fails here, so every evaluator sweep chains them in.
pub const ODD_WIDTHS: [usize; 3] = [3, 5, 7];

/// Corner raw patterns for a `bits`-wide operand, deduplicated and masked to the width.
///
/// Host words rather than [`IntBits`], because the list is sorted to deduplicate and a pattern has
/// no ordering. The sweeps using them are host-level evaluators, which build a pattern only where
/// they call the model.
#[must_use]
pub fn values(bits: usize) -> Vec<u128> {
    assert!((1..=HOST_CORNER_BITS).contains(&bits));
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

/// Shift amounts to try against a `bits`-wide value, as raw `bits`-wide patterns.
#[must_use]
pub fn shift_amounts(bits: usize) -> Vec<u128> {
    assert!((1..=HOST_CORNER_BITS).contains(&bits));
    let m = mask(bits);
    let mut out = vec![0, 1, 63, 64, 65, 127, 128, m];

    for around in [bits, bits / 2] {
        for delta in [0usize, 1, 2] {
            out.push((around + delta) as u128);
            out.push((around.saturating_sub(delta)) as u128);
        }
    }

    // A negative amount, which reads as a huge magnitude and must be rejected for that reason.
    if bits >= 2 {
        out.push(1u128 << (bits - 1));
    }

    out.iter_mut().for_each(|v| *v &= m);
    out.sort_unstable();
    out.dedup();
    out
}

/// The reads worth taking of a `bits`-wide container by a spread or an unspread, the narrowest of
/// which is `least`: all of it, one bit short, half, `least` itself, and either side of the first
/// limb boundary, where a limb-wise lowering splits a read across two limbs.
#[must_use]
pub fn spread_reads(bits: usize, least: usize) -> Vec<usize> {
    let mut reads = vec![
        bits,
        bits.saturating_sub(1),
        bits / 2,
        least,
        HOST_LIMB_BITS,
        HOST_LIMB_BITS + 1,
    ];
    reads.retain(|&read| (least..=bits).contains(&read));
    reads.sort_unstable();
    reads.dedup();
    reads
}

/// The widths every narrow sweep covers, under either reading: [`WIDTHS`] **plus
/// [`ODD_WIDTHS`]**, joined here so no sweep can leave out the odd widths.
#[must_use]
pub fn widths() -> &'static [usize] {
    static WIDTHS_WITH_ODD: LazyLock<Vec<usize>> = LazyLock::new(|| {
        let mut out: Vec<usize> = WIDTHS.iter().copied().chain(ODD_WIDTHS).collect();
        out.sort_unstable();
        out.dedup();
        out
    });

    &WIDTHS_WITH_ODD
}

// THE WIDE CORNER SET
// ================================================================================================
//
// Everything below carries [`IntBits`] end to end, the only way to name a corner of a 16384-bit
// type.
//
// It is a **separate** set rather than wider entries in [`WIDTHS`]: all nine registered evaluators
// drive their sweeps off [`widths`], so a wide entry would ask each to answer at 16384 bits whether
// or not its lane has a wide body. A sweep opts in by naming [`WIDE_WIDTHS`].
//
// The interval domain cannot opt in: its relation quantifies over the concretization of its
// ranges, which is not enumerable at 16384 bits.

/// The widths a wide sweep covers.
///
/// Few on purpose, as a corner sweep is quadratic in full-width operands, so each width asks a
/// different question:
///
/// - `129`: one past the narrow/wide threshold, the narrowest width that goes limb-wise.
/// - `192`: exactly three full limbs, with no `bits % 64` special case.
/// - `256`: the width the two-limb product lane would reach for.
/// - `1000`: neither a multiple of the limb width nor a power of two.
/// - **`16383` and `16384` must both stay**: the partial and the full top limb at the cap. `16384`
///   has no `bits % HOST_LIMB_BITS` special case and `16383` exercises all of them.
pub const WIDE_WIDTHS: [usize; 6] = [129, 192, 256, 1000, 16383, 16384];

/// Corner patterns for a `bits`-wide operand, deduplicated.
///
/// The wide counterpart of [`values`], unsorted: a pattern has two orderings (signed and unsigned)
/// and this does not pick one, so deduplication scans the short list.
///
/// Covers what [`values`] covers, less its three fixed constants (`40`, `100`, `200`), **plus the
/// limb boundaries**: the top of the low limb, the bottom of the second, and the same pair at the
/// top limb. A carry chain, mask or sign extension off by one limb is wrong exactly there.
#[must_use]
pub fn wide_values(bits: usize) -> Vec<IntBits> {
    assert!(
        (1..=crate::MAX_BITS).contains(&bits),
        "width {bits} is outside 1..={}",
        crate::MAX_BITS
    );

    let one = IntBits::from_u128(bits, 1);
    let pow2 = |k: usize| one.shifted_left(k);
    // `2^k - 1` at this width: all ones in the low `k` bits, zero above.
    let below_pow2 = |k: usize| IntBits::all_ones(k).cast(bits);

    let mut out = vec![
        IntBits::zero(bits),
        one.clone(),
        IntBits::from_u128(bits, 2),
        IntBits::from_u128(bits, 3),
        // `-1`, `-2` and `-3` as raw patterns: `xor` off the low bits of all-ones subtracts,
        // because the bits being cleared are set.
        IntBits::all_ones(bits),
        IntBits::all_ones(bits).xor(&one),
        IntBits::all_ones(bits).xor(&IntBits::from_u128(bits, 2)),
    ];

    if bits >= 2 {
        // `MIN_S`, `MAX_S` and one step inside each.
        let sign_bit = pow2(bits - 1);
        out.push(sign_bit.clone());
        out.push(sign_bit.or(&one));
        out.push(below_pow2(bits - 1));
        out.push(below_pow2(bits - 1).xor(&one));

        // `2^(bits/2)`, whose square is `2^bits` at an even width: the product one past the top.
        out.push(pow2(bits / 2));
    }

    // The limb boundaries, which is what makes this set wide rather than merely large. `k` is the
    // limb count, so `(k - 1) * HOST_LIMB_BITS` is where the top limb starts.
    let top_limb_start = (IntBits::limbs_for_bits(bits) - 1) * HOST_LIMB_BITS;
    for boundary in [HOST_LIMB_BITS, top_limb_start, bits] {
        if boundary > 0 && boundary <= bits {
            out.push(below_pow2(boundary));
            if boundary < bits {
                out.push(pow2(boundary));
                out.push(pow2(boundary).xor(&one));
            }
        }
    }

    // Alternating patterns catch a mask applied at the wrong width, which uniform values cannot,
    // and at a wide width they also catch one applied to the wrong _limb_.
    let limbs = IntBits::limbs_for_bits(bits);
    out.push(IntBits::from_limbs(
        bits,
        &vec![0x5555_5555_5555_5555u64; limbs],
    ));
    out.push(IntBits::from_limbs(
        bits,
        &vec![0xAAAA_AAAA_AAAA_AAAAu64; limbs],
    ));

    dedup_patterns(out)
}

/// Shift amounts to try against a `bits`-wide value, as `bits`-wide patterns.
///
/// The wide counterpart of [`shift_amounts`]: amounts around `bits`, as there, plus amounts around
/// the **limb** size. A wide shift moves `q` whole limbs plus `r` bits, so `HOST_LIMB_BITS` and its
/// neighbors are where `r` is zero and there is no split within a limb.
#[must_use]
pub fn wide_shift_amounts(bits: usize) -> Vec<IntBits> {
    assert!(
        (1..=crate::MAX_BITS).contains(&bits),
        "width {bits} is outside 1..={}",
        crate::MAX_BITS
    );

    let mut out = vec![IntBits::zero(bits), IntBits::from_u128(bits, 1)];

    for around in [HOST_LIMB_BITS, bits / 2, bits] {
        for delta in [0usize, 1, 2] {
            out.push(IntBits::from_biguint(
                bits,
                &BigUint::from(around.saturating_add(delta)),
            ));
            out.push(IntBits::from_biguint(
                bits,
                &BigUint::from(around.saturating_sub(delta)),
            ));
        }
    }

    // The whole amount range as a magnitude, and a negative one, which reads as a huge magnitude
    // and must be rejected for that reason.
    out.push(IntBits::all_ones(bits));
    if bits >= 2 {
        out.push(IntBits::from_u128(bits, 1).shifted_left(bits - 1));
    }

    dedup_patterns(out)
}

/// The `(lhs, rhs)` corner lists a wide sweep runs over for one operation: [`wide_values`] on both
/// sides, or [`wide_shift_amounts`] on the right of a shift. As a shift amount every corner value
/// is far past the width, so it would test only the rejection.
#[must_use]
pub fn wide_operands(op: IntOp, bits: usize) -> (Vec<IntBits>, Vec<IntBits>) {
    let values = wide_values(bits);
    let rhs = if op.is_shift() {
        wide_shift_amounts(bits)
    } else {
        values.clone()
    };
    (values, rhs)
}

/// Remove duplicates from a corner list, preserving order. Quadratic, as the lists are a few dozen
/// entries and a pattern has no ordering to sort by.
fn dedup_patterns(patterns: Vec<IntBits>) -> Vec<IntBits> {
    let mut out: Vec<IntBits> = Vec::with_capacity(patterns.len());
    for p in patterns {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both `16383` and `16384` are present: `16384` has no `bits % HOST_LIMB_BITS` special case
    /// and `16383` exercises all of them, so either alone can hide a top-limb bug.
    #[test]
    fn the_cap_appears_as_both_a_full_and_a_partial_top_limb() {
        assert!(WIDE_WIDTHS.contains(&16384) && WIDE_WIDTHS.contains(&16383));
        assert_eq!(
            16384 % HOST_LIMB_BITS,
            0,
            "16384 is a whole number of limbs"
        );
        assert_ne!(16383 % HOST_LIMB_BITS, 0, "16383 is not");
        assert_eq!(
            IntBits::limbs_for_bits(16383),
            IntBits::limbs_for_bits(16384),
            "and they occupy the same number of limbs, so the difference is the mask alone"
        );
    }

    /// Every corner is at its requested width with no bit above it. A corner built by shifting or
    /// complementing can land outside the width, which the host-typed generators cannot.
    #[test]
    fn every_wide_corner_is_at_its_declared_width() {
        for &bits in &WIDE_WIDTHS {
            for p in wide_values(bits) {
                assert_eq!(p.bits(), bits);
                assert_eq!(p.limb_count(), IntBits::limbs_for_bits(bits));
            }
            for a in wide_shift_amounts(bits) {
                assert_eq!(a.bits(), bits);
            }
        }
    }

    /// The corners that make this set wide rather than merely large.
    #[test]
    fn the_limb_boundaries_are_all_present() {
        let bits = 1000;
        let corners = wide_values(bits);
        let one = IntBits::from_u128(bits, 1);
        let top_limb_start = (IntBits::limbs_for_bits(bits) - 1) * HOST_LIMB_BITS;

        for k in [HOST_LIMB_BITS, top_limb_start] {
            assert!(
                corners.contains(&IntBits::all_ones(k).cast(bits)),
                "the top of the limb below bit {k} is missing"
            );
            assert!(
                corners.contains(&one.shifted_left(k)),
                "the bottom of the limb at bit {k} is missing"
            );
        }

        // And the value's top, which is `-1` under the other reading.
        assert!(corners.contains(&IntBits::all_ones(bits)));
        assert!(corners.contains(&IntBits::zero(bits)));
    }

    /// Every wide width is inside the model's domain, so no sweep driven off the set is quietly
    /// narrower than it reads.
    #[test]
    fn every_wide_width_is_inside_the_model() {
        assert!(WIDE_WIDTHS.iter().all(|&b| b <= crate::MAX_BITS));
    }

    /// The narrow generators are untouched by any of this, which is what keeps the split honest.
    #[test]
    fn the_narrow_generators_still_refuse_a_wide_width() {
        assert!(WIDTHS.iter().all(|&b| b <= HOST_CORNER_BITS));
        assert!(std::panic::catch_unwind(|| values(129)).is_err());
        assert!(std::panic::catch_unwind(|| shift_amounts(129)).is_err());
    }

    /// The model answers at every wide width, the precondition for any sweep using them. One pair
    /// per width: this checks admissibility, not arithmetic.
    #[test]
    fn the_model_accepts_every_wide_width() {
        for &bits in &WIDE_WIDTHS {
            // Built here, not indexed out of `wide_values`, whose order is an implementation
            // detail.
            let (a, b) = (IntBits::from_u128(bits, 1), IntBits::from_u128(bits, 2));
            assert_eq!(
                crate::eval(crate::IntOp::UAdd, &a, &b).value(),
                Some(IntBits::from_u128(bits, 3)),
                "1 + 2 at {bits} bits"
            );

            // And that the generator really produces corners at that width for a sweep to use.
            assert!(!wide_values(bits).is_empty());
        }
    }
}
