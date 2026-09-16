//! The corner values every conformance sweep runs over.
//!
//! These are deliberately the _known_ corners: sign boundaries, `INT_MIN`, `-1`, the width and
//! the width ± 1 as shift amounts. Random operand generation essentially never produces them, which
//! is why this list is the backbone and `proptest` is the layer on top rather than the other way
//! round.

use std::sync::LazyLock;

use num_bigint::BigUint;

use crate::{
    IntBits, IntOp, MAX_LOWERED_SIGNED_BITS,
    int_bits::{HOST_LIMB_BITS, HOST_WORD_BITS},
    mask,
};

/// The widest pattern the generators below can express.
///
/// They are host-typed by design — `Vec<u128>`, sorted and deduplicated, per [`values`]'s doc —
/// so this is a property of the return type and deliberately **not** [`MAX_BITS`](crate::MAX_BITS),
/// which is far above it. Written as the cap, raising it would loosen the asserts below
/// without widening anything they guard, and a `values(16384)` would sweep a 128-bit corner set
/// at a 16384-bit width: passing, while checking almost nothing.
///
/// The wide half of the corner set is a separate generator returning [`IntBits`](crate::IntBits),
/// which each evaluator's sweep opts into as its own lane learns the width.
const HOST_CORNER_BITS: usize = HOST_WORD_BITS;

/// The widths every sweep covers.
///
/// `1` is `bool`, and a corner in its own right: it is the only width where the sole negative value
/// is `1`, and where `bits - 1` is `0` so every shift amount masks away.
pub const WIDTHS: [usize; 6] = [1, 8, 16, 32, 64, 128];

/// The widths a signed sweep covers — [`WIDTHS`] without the ones no signed operation may touch.
pub const SIGNED_WIDTHS: [usize; 5] = [1, 8, 16, 32, 64];

/// Widths small enough to sweep _every_ operand pair at.
///
/// `8` gives 65 536 pairs per operation, which is a second's work for the whole matrix. Anything
/// wider has to fall back to the corner set.
pub const EXHAUSTIVE_WIDTHS: [usize; 3] = [1, 4, 8];

/// Widths that are not powers of two.
///
/// Kept as a named set because they are the ones that catch a width assumption, and every
/// evaluator sweep now chains them in: a mask by `bits - 1` passes at every width in [`WIDTHS`]
/// and fails here.
pub const ODD_WIDTHS: [usize; 3] = [3, 5, 7];

/// Corner raw patterns for a `bits`-wide operand, deduplicated and masked to the width.
///
/// Deliberately host words rather than [`IntBits`](crate::IntBits) as these are a _corpus_, sorted
/// and deduplicated, and a pattern has no ordering to sort by. The sweeps that use them are
/// host-level evaluators too, so they build a pattern at the point they call the model and nowhere
/// else.
///
/// Covers, in both readings: zero and the small values, the unsigned top, the signed boundary pair
/// and its neighbours, `-1` and `-2`, the powers of two that sit at the width's edges and middle,
/// the two alternating patterns, and the two operands the existing corpus tests use for the case
/// where the readings disagree.
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

/// Shift amounts to try against a `bits`-wide value, as raw patterns at `rhs_bits`.
///
/// The amounts around `bits` are the point: `bits - 1` is the largest legal one, `bits` is the
/// smallest rejected one, and an evaluator that gets the boundary off by one is wrong on exactly
/// those two. The host widths (63, 64, 65, 127, 128) are here because several evaluators reach for
/// a `u64` or `u128` shift internally and inherit _its_ masking rather than the operand's.
#[must_use]
pub fn shift_amounts(bits: usize, rhs_bits: usize) -> Vec<u128> {
    assert!((1..=HOST_CORNER_BITS).contains(&bits) && (1..=HOST_CORNER_BITS).contains(&rhs_bits));
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
    let mut out = Vec::new();
    for &bits in widths_for(signed) {
        for &rhs_bits in &[bits, 8, 32, 64, HOST_CORNER_BITS] {
            out.push((bits, rhs_bits));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The widths a sweep should use for `sign`.
///
/// This is [`WIDTHS`] (or [`SIGNED_WIDTHS`]) **plus [`ODD_WIDTHS`]**. The union lives here rather
/// than in each sweep on purpose to avoid width assumptions.
///
/// Built once per reading rather than per call, and borrowed rather than cloned: every caller is a
/// sweep that asks for this from inside a loop.
#[must_use]
pub fn widths_for(signed: bool) -> &'static [usize] {
    static UNSIGNED: LazyLock<Vec<usize>> = LazyLock::new(|| union_with_odd(false));
    static SIGNED: LazyLock<Vec<usize>> = LazyLock::new(|| union_with_odd(true));

    if signed { &SIGNED } else { &UNSIGNED }
}

/// The body of [`widths_for`], run once per reading.
fn union_with_odd(signed: bool) -> Vec<usize> {
    let base: &[usize] = if signed { &SIGNED_WIDTHS } else { &WIDTHS };
    let mut out: Vec<usize> = base
        .iter()
        .copied()
        .chain(ODD_WIDTHS)
        .filter(|bits| !signed || signed_width_ok(*bits))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Whether a width is one a signed operation may be _lowered_ at.
///
/// Intentionally mirrors [`MAX_LOWERED_SIGNED_BITS`], so a sweep covers only the widths each
/// evaluator can currently answer for.
#[must_use]
pub fn signed_width_ok(bits: usize) -> bool {
    (1..=MAX_LOWERED_SIGNED_BITS).contains(&bits)
}

// THE WIDE CORNER SET
// ================================================================================================
//
// Everything above is host-typed and stops at [`HOST_CORNER_BITS`]. Everything below carries
// [`IntBits`] end to end, which is the only way to name a corner of a 16384-bit type at all.
//
// It is a **separate** set rather than wider entries in [`WIDTHS`], and that is the load-bearing
// decision here. All nine registered evaluators drive their sweeps off [`widths_for`] and most
// take it unfiltered, so a wide entry there would ask every evaluator to answer at 16384 bits at
// once — here, in the unit that only moves a cap, before any lane can. Opting in per evaluator
// makes "turn my sweep's wide set on" the first act of each later unit, which is precisely the
// demonstration that unit owes.
//
// One registered evaluator cannot opt in at all: the interval domain's relation quantifies over the
// _concretisation_ of its ranges rather than over corners, and the concretisation of a 16384-bit
// interval is not enumerable.

/// The widths a wide sweep covers.
///
/// Small on purpose: a corner sweep is quadratic in the corner count and each operand is a
/// `BigUint` of the full width, so this is the set of widths that ask a _different question_
/// rather than a sample of the range.
///
/// - `129` is one bit past the narrow/wide threshold, the narrowest width that must go limb-wise.
/// - `192` is exactly three limbs, where the top limb is full and every `bits % 64` special case
///   vanishes.
/// - `256` is the width the two-limb product lane would reach for, and the first that is not also
///   a power-of-two multiple of the limb.
/// - `1000` is not a multiple of the limb width and not a power of two, which is where a lowering
///   that assumed either is wrong.
/// - **`16383` and `16384` are both here and neither may be dropped.** They are the partial and
///   the full top limb at the cap: `16384` is exactly 256 limbs, so every `bits % HOST_LIMB_BITS`
///   special case disappears, and `16383` is the one that exercises all of them. Testing either
///   alone is the single easiest way to ship a top-limb bug.
pub const WIDE_WIDTHS: [usize; 6] = [129, 192, 256, 1000, 16383, 16384];

/// The wide widths a sweep should use for `sign`.
///
/// Filtered by [`signed_width_ok`], so today this is the whole of [`WIDE_WIDTHS`] for an unsigned
/// sweep and **empty** for a signed one — no lowering reads a signed pattern above one host limb.
/// The signed wide sweeps therefore exist, compile and run zero cases, and P5's signed unit turns
/// every one of them on by moving [`MAX_LOWERED_SIGNED_BITS`] rather than by editing nine sweeps.
#[must_use]
pub fn wide_widths_for(signed: bool) -> Vec<usize> {
    WIDE_WIDTHS
        .into_iter()
        .filter(|bits| !signed || signed_width_ok(*bits))
        .collect()
}

/// Corner patterns for a `bits`-wide operand, deduplicated.
///
/// The wide counterpart of [`values`], and deliberately not sorted: two patterns have two
/// orderings and this type refuses to pick one, which is the same reason [`values`] is host-typed
/// and this is not. Dedup is therefore linear scan over a list this short rather than
/// `sort`+`dedup`.
///
/// Covers what [`values`] covers — zero and the small values, the unsigned top, the signed
/// boundary pair and its neighbours, `-1` and `-2`, the alternating patterns — **plus the limb
/// boundaries**, which are the corners that do not exist at a narrow width: the top of the low
/// limb, the bottom of the second, and the same pair at the top limb. A carry chain, a mask or a
/// sign extension that is off by one limb is wrong on exactly those and right everywhere else.
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

/// Shift amounts to try against a `bits`-wide value, as patterns at `rhs_bits`.
///
/// The wide counterpart of [`shift_amounts`]. The amounts around `bits` are the point for the same
/// reason they are there, and the **limb** boundaries join them: a wide shift decomposes into
/// `q` whole limbs plus `r` bits, so `HOST_LIMB_BITS` and its neighbours are where `r` is zero and
/// an intra-limb split has nothing to split.
#[must_use]
pub fn wide_shift_amounts(bits: usize, rhs_bits: usize) -> Vec<IntBits> {
    assert!(
        (1..=crate::MAX_BITS).contains(&bits) && (1..=crate::MAX_BITS).contains(&rhs_bits),
        "widths {bits}/{rhs_bits} are outside 1..={}",
        crate::MAX_BITS
    );

    let mut out = vec![IntBits::zero(rhs_bits), IntBits::from_u128(rhs_bits, 1)];

    for around in [HOST_LIMB_BITS, bits / 2, bits] {
        for delta in [0usize, 1, 2] {
            out.push(IntBits::from_biguint(
                rhs_bits,
                &BigUint::from(around.saturating_add(delta)),
            ));
            out.push(IntBits::from_biguint(
                rhs_bits,
                &BigUint::from(around.saturating_sub(delta)),
            ));
        }
    }

    // The whole amount range as a magnitude, and a negative one, which reads as a huge magnitude
    // and must be rejected for that reason.
    out.push(IntBits::all_ones(rhs_bits));
    if rhs_bits >= 2 {
        out.push(IntBits::from_u128(rhs_bits, 1).shifted_left(rhs_bits - 1));
    }

    dedup_patterns(out)
}

/// The `(lhs, rhs)` corner lists a wide sweep should run over for one operation.
///
/// Every wide sweep wants the same thing and wants it for the same reason: [`wide_values`] on the
/// left, and on the right either [`wide_values`] again or [`wide_shift_amounts`] for a shift,
/// because a corner _value_ at a wide width is a magnitude so far past the width that every one of
/// them is rejected alike, and a sweep built from those would check the rejection path a few
/// hundred times and the shifting path never.
#[must_use]
pub fn wide_operands(op: IntOp, bits: usize) -> (Vec<IntBits>, Vec<IntBits>) {
    let values = wide_values(bits);
    let rhs = if op.is_shift() {
        wide_shift_amounts(bits, bits)
    } else {
        values.clone()
    };
    (values, rhs)
}

/// Remove duplicates from a corner list, preserving order.
///
/// Quadratic, and deliberately: the lists above are a few dozen entries and a pattern has no
/// ordering to sort by, so the alternative would be inventing one here purely to deduplicate.
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

    /// The one property the whole wide set exists for.
    ///
    /// `16384` is exactly 256 limbs, so every `bits % HOST_LIMB_BITS` special case in every
    /// lowering disappears at it; `16383` is the width that exercises all of them. A sweep run at
    /// either alone is the easiest way to ship a top-limb bug, so this pins that both are present
    /// and that they really do differ in the way that matters.
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

    /// Every corner is at the width it was asked for, and none carries a bit above it.
    ///
    /// The failure this catches is the one the host-typed generators cannot have: a corner built
    /// by shifting or complementing that lands outside the width and is silently normalised, or
    /// worse, is not.
    #[test]
    fn every_wide_corner_is_at_its_declared_width() {
        for &bits in &WIDE_WIDTHS {
            for p in wide_values(bits) {
                assert_eq!(p.bits(), bits);
                assert_eq!(p.limb_count(), IntBits::limbs_for_bits(bits));
            }
            for a in wide_shift_amounts(bits, bits) {
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

        // And the value's own top, which is `-1` under the other reading.
        assert!(corners.contains(&IntBits::all_ones(bits)));
        assert!(corners.contains(&IntBits::zero(bits)));
    }

    /// The signed wide sweeps are switched off by one constant, not by nine edits.
    #[test]
    fn the_wide_signed_sweep_is_empty_until_the_frontier_moves() {
        assert!(
            wide_widths_for(true).is_empty(),
            "no lowering reads a signed pattern above one host limb yet"
        );
        assert_eq!(wide_widths_for(false).len(), WIDE_WIDTHS.len());

        // What P5's signed unit changes, stated as the thing it changes: every wide width becomes
        // signed-legal the moment `MAX_LOWERED_SIGNED_BITS` reaches the cap, with no edit here.
        assert!(
            WIDE_WIDTHS.iter().all(|&b| b <= crate::MAX_BITS),
            "a wide width outside the model's domain would stay filtered out even then"
        );
    }

    /// The narrow generators are untouched by any of this, which is what keeps the split honest.
    #[test]
    fn the_narrow_generators_still_refuse_a_wide_width() {
        assert!(WIDTHS.iter().all(|&b| b <= HOST_CORNER_BITS));
        assert!(std::panic::catch_unwind(|| values(129)).is_err());
        assert!(std::panic::catch_unwind(|| shift_amounts(129, 129)).is_err());
    }

    /// The model answers at every wide width, which is the precondition for any sweep using these.
    ///
    /// Cheap on purpose — one operand pair per width, not the corner matrix — because what is
    /// being checked is that the widths are admissible at all, not that the arithmetic is right.
    #[test]
    fn the_model_accepts_every_wide_width() {
        for &bits in &WIDE_WIDTHS {
            // Built here rather than indexed out of `wide_values`, whose order is an implementation
            // detail of the generator: what this asks is whether the model answers at the width,
            // not what the corner list happens to hold at a position.
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
