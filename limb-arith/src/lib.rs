//! Wrapping arithmetic, and the spread and unspread bit rearrangements, on buffers of little-endian
//! `u64` limbs.
//!
//! Shared by the evaluators that compute wide integers in limbs: the VM and the WASM runtime's
//! helpers.
//!
//! This crate is `no_std` and does not allocate.

#![no_std]

// SUMS AND PRODUCTS
// ================================================================================================

/// `a + b` written into `res`, all arguments of the same length.
pub fn add(res: &mut [u64], a: &[u64], b: &[u64]) {
    let mut carry = false;
    for i in 0..res.len() {
        let (sum, overflowed) = a[i].overflowing_add(b[i]);
        let (sum, carried) = sum.overflowing_add(u64::from(carry));
        res[i] = sum;
        carry = overflowed | carried;
    }
}

/// `a - b` written into `res`, all arguments of the same length.
pub fn sub(res: &mut [u64], a: &[u64], b: &[u64]) {
    let mut borrow = false;
    for i in 0..res.len() {
        let (difference, underflowed) = a[i].overflowing_sub(b[i]);
        let (difference, borrowed) = difference.overflowing_sub(u64::from(borrow));
        res[i] = difference;
        borrow = underflowed | borrowed;
    }
}

/// `a * b` written into `res`, all arguments of the same length.
///
/// Schoolbook, accumulating straight into the result rather than into a `2k`-limb product and
/// truncating after: the columns at or above `k` are exactly the ones the wrap discards, so the
/// high half is never computed and needs no scratch buffer. The carry leaving column `k-1` is
/// dropped for the same reason.
///
/// The column accumulator needs no wrapping arithmetic: a limb product plus two limb-sized addends
/// is at most `(2^64 - 1)^2 + 2·(2^64 - 1)`, which is `2^128 - 1` exactly.
pub fn mul(res: &mut [u64], a: &[u64], b: &[u64]) {
    let k = res.len();
    res.fill(0);
    for i in 0..k {
        let mut carry = 0u64;
        for j in 0..(k - i) {
            let column =
                u128::from(a[i]) * u128::from(b[j]) + u128::from(res[i + j]) + u128::from(carry);
            res[i + j] = column as u64;
            carry = (column >> 64) as u64;
        }
    }
}

/// Replace `value` with its two's complement negation, `!value + 1`.
pub fn negate(value: &mut [u64]) {
    let mut carry = true;
    for limb in value.iter_mut() {
        let (complemented, carried) = (!*limb).overflowing_add(u64::from(carry));
        *limb = complemented;
        carry = carried;
    }
}

// DIVISION
// ================================================================================================

/// The index one past the highest non-zero limb, so zero answers zero.
#[must_use]
pub fn significant_limbs(value: &[u64]) -> usize {
    value
        .iter()
        .rposition(|&limb| limb != 0)
        .map_or(0, |i| i + 1)
}

/// `a / divisor` written into `quotient`, which is at least as long as `a`, and the remainder
/// returned.
///
/// Schoolbook division by one limb, top limb first. `u128` division supplies the
/// two-limb-by-one-limb step a 64-bit host does not have as an operator, and the running remainder
/// stays below `divisor`, so each quotient limb fits a limb.
///
/// # Panics
///
/// If `divisor` is zero.
pub fn divide_by_limb(quotient: &mut [u64], a: &[u64], divisor: u64) -> u64 {
    let divisor = u128::from(divisor);
    let mut rest = 0u128;
    for i in (0..a.len()).rev() {
        let dividend = (rest << 64) | u128::from(a[i]);
        quotient[i] = (dividend / divisor) as u64;
        rest = dividend % divisor;
    }
    rest as u64
}

/// `a / b` and `a % b` for a divisor of exactly `b.len()` significant limbs, written into
/// `quotient` and `remainder`, which the caller has zeroed and which are at least as long as `a`.
///
/// `divisor` and `dividend` are working buffers of at least `b.len()` and `a.len() + 1` limbs,
/// whose contents on entry do not matter.
///
/// The estimate `qhat` for each quotient limb comes from dividing the top two limbs of the running
/// dividend by the top limb of the divisor, which is why the divisor is first shifted left until
/// its top bit is set: the estimate is then at most two above the true limb (Knuth's Theorem B).
/// The test against the divisor's second limb brings it to at most one above, and the add-back
/// after the subtraction repairs that last one. `u128` arithmetic supplies the two-limb steps a
/// 64-bit host does not have as operators.
pub fn knuth_divide(
    quotient: &mut [u64],
    remainder: &mut [u64],
    a: &[u64],
    b: &[u64],
    divisor: &mut [u64],
    dividend: &mut [u64],
) {
    const BASE: u128 = 1 << 64;
    let n = b.len();
    debug_assert!(n >= 2, "a divisor of {n} limbs takes `divide_by_limb`");
    debug_assert_eq!(significant_limbs(b), n, "the divisor's top limb is zero");

    let m = significant_limbs(a);
    if m < n {
        remainder[..a.len()].copy_from_slice(a);
        return;
    }

    // Normalize so the divisor's top bit is set. The same shift on the dividend leaves the quotient
    // unchanged and scales the remainder, which is undone at the end.
    let shift = b[n - 1].leading_zeros();
    let divisor = &mut divisor[..n];
    shift_left_into(divisor, b, shift);

    // One limb wider than the dividend: the shift can carry out of the top, and the algorithm reads
    // `dividend[j + n]` at the highest `j`.
    let dividend = &mut dividend[..=m];
    shift_left_into(dividend, &a[..m], shift);

    for j in (0..=(m - n)).rev() {
        let top = (u128::from(dividend[j + n]) << 64) | u128::from(dividend[j + n - 1]);
        let mut estimate = top / u128::from(divisor[n - 1]);
        let mut rest = top % u128::from(divisor[n - 1]);

        // Bring the estimate down to at most one above the true limb. The first disjunct is checked
        // first so the product below is only ever formed for an estimate that fits a limb.
        while estimate >= BASE
            || estimate * u128::from(divisor[n - 2])
                > (rest << 64) | u128::from(dividend[j + n - 2])
        {
            estimate -= 1;
            rest += u128::from(divisor[n - 1]);
            if rest >= BASE {
                break;
            }
        }

        // Subtract `estimate * divisor` from the window, keeping a signed borrow: the estimate may
        // still be one too large, and that is the case the add-back below repairs.
        let mut borrow = 0i128;
        for i in 0..n {
            let product = estimate * u128::from(divisor[i]);
            let column = i128::from(dividend[i + j]) - borrow - i128::from(product as u64);
            dividend[i + j] = column as u64;
            borrow = (product >> 64) as i128 - (column >> 64);
        }
        let column = i128::from(dividend[j + n]) - borrow;
        dividend[j + n] = column as u64;

        quotient[j] = estimate as u64;
        if column < 0 {
            quotient[j] -= 1;
            let mut carry = false;
            for i in 0..n {
                let (sum, overflowed) = dividend[i + j].overflowing_add(divisor[i]);
                let (sum, carried) = sum.overflowing_add(u64::from(carry));
                dividend[i + j] = sum;
                carry = overflowed | carried;
            }
            dividend[j + n] = dividend[j + n].wrapping_add(u64::from(carry));
        }
    }

    // Undo the normalising shift to recover the true remainder.
    shift_right_into(&mut remainder[..n], &dividend[..n], shift);
}

/// `source << shift` written into `target`, where `shift` is under 64 and `target` may be longer.
fn shift_left_into(target: &mut [u64], source: &[u64], shift: u32) {
    for i in (0..target.len()).rev() {
        let low = source.get(i).copied().unwrap_or(0);
        let carried = if shift > 0 && i > 0 {
            source.get(i - 1).copied().unwrap_or(0) >> (64 - shift)
        } else {
            0
        };
        target[i] = (low << shift) | carried;
    }
}

/// `source >> shift` written into `target`, where `shift` is under 64 and the other two have the
/// same length.
fn shift_right_into(target: &mut [u64], source: &[u64], shift: u32) {
    for i in 0..target.len() {
        let high = source[i];
        let carried = if shift > 0 && i + 1 < source.len() {
            source[i + 1] << (64 - shift)
        } else {
            0
        };
        target[i] = (high >> shift) | carried;
    }
}

// SPREADS
// ================================================================================================

/// `v` with bit `i` moved to bit `2i` and zeros between, which is the spread of one host half-word.
#[must_use]
pub fn spread_u32_to_u64(v: u32) -> u64 {
    let mut x = u64::from(v);
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    (x | (x << 1)) & 0x5555_5555_5555_5555
}

/// The odd- and even-indexed bits of `v` as `(odd, even)`, each moved down to be contiguous, which
/// is the unspread of one host word.
#[must_use]
pub fn unspread_u64_to_u32(v: u64) -> (u32, u32) {
    (compact_even_bits(v >> 1), compact_even_bits(v))
}

/// The even-indexed bits of `x`, bit `2i` moved to bit `i`.
fn compact_even_bits(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x as u32
}

/// Limb `index` of `a` with every bit at or above `bits` cleared, so that a spread or unspread
/// reads only the bits it names.
fn read_limb(a: &[u64], index: usize, bits: usize) -> u64 {
    match bits - 64 * index {
        rest if rest >= 64 => a[index],
        rest => a[index] & ((1 << rest) - 1),
    }
}

/// The low `bits` bits of `a` spread into `res`, bit `i` to bit `2i`, with every other bit of `res`
/// cleared.
///
/// `a` holds at least the limbs `bits` reaches and `res` at least the limbs `2 * bits` does; the
/// two need not be the same length, and anything past those is read as zero or written as zero.
pub fn spread(res: &mut [u64], a: &[u64], bits: usize) {
    res.fill(0);
    for index in 0..bits.div_ceil(64) {
        let limb = read_limb(a, index, bits);
        res[2 * index] = spread_u32_to_u64(limb as u32);
        let high = spread_u32_to_u64((limb >> 32) as u32);
        // A top limb of 32 bits or fewer has an empty high half, and `res` may stop short of it.
        if let Some(slot) = res.get_mut(2 * index + 1) {
            *slot = high;
        } else {
            debug_assert_eq!(
                high,
                0,
                "a spread of {bits} bits does not fit {} limbs",
                res.len()
            );
        }
    }
}

/// The low `bits` bits of `a` separated into `odd` and `even`, bit `2i + 1` to bit `i` of `odd` and
/// bit `2i` to bit `i` of `even`, with every other bit of both cleared.
///
/// `a` holds at least the limbs `bits` reaches, and each stream at least the limbs its own
/// `floor(bits / 2)` or `ceil(bits / 2)` bits do.
pub fn unspread(odd: &mut [u64], even: &mut [u64], a: &[u64], bits: usize) {
    odd.fill(0);
    even.fill(0);
    for index in 0..bits.div_ceil(64) {
        let (odd_half, even_half) = unspread_u64_to_u32(read_limb(a, index, bits));
        let shift = 32 * (index % 2);
        for (stream, half) in [(&mut *odd, odd_half), (&mut *even, even_half)] {
            // The odd stream is a bit shorter at an odd width, and may stop short of a top limb
            // whose only bit is an even one.
            if let Some(slot) = stream.get_mut(index / 2) {
                *slot |= u64::from(half) << shift;
            } else {
                debug_assert_eq!(
                    half, 0,
                    "an unspread of {bits} bits does not fit its streams"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread unspreads back to the bits it read, with nothing in the odd stream, at every width
    /// across three limbs, including the reads that end inside a limb's low half.
    ///
    /// The sweeps against the model are the VM's and the runtime's; this pins the shared body on
    /// its own, including that bits above the read are cleared rather than carried through. Every
    /// output starts as garbage, as a VM frame cell does, so a limb the body failed to clear shows.
    #[test]
    fn an_unspread_undoes_a_spread_at_every_width() {
        const GARBAGE: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        let a = [0xDEAD_BEEF_F00D_CAFE, u64::MAX, 0x0123_4567_89AB_CDEF];
        for bits in 1..=192 {
            let mut spread_out = [GARBAGE; 6];
            spread(&mut spread_out, &a, bits);

            let (mut odd, mut even) = ([GARBAGE; 3], [GARBAGE; 3]);
            unspread(&mut odd, &mut even, &spread_out, 2 * bits);
            assert_eq!(odd, [0; 3], "{bits} bits");
            let mut want = [0u64; 3];
            for (index, slot) in want.iter_mut().enumerate().take(bits.div_ceil(64)) {
                *slot = read_limb(&a, index, bits);
            }
            assert_eq!(even, want, "{bits} bits");
        }
    }

    /// Assert `quotient·b + remainder == a` with `remainder < b`, for the division of `a` by `b`.
    fn assert_divides(a: &[u64], b: &[u64]) {
        let n = significant_limbs(b);
        let (mut quotient, mut remainder) = ([0u64; 8], [0u64; 8]);
        let (mut divisor, mut dividend) = ([0u64; 8], [0u64; 9]);
        let (quotient, remainder) = (&mut quotient[..a.len()], &mut remainder[..a.len()]);
        knuth_divide(quotient, remainder, a, &b[..n], &mut divisor, &mut dividend);

        let mut check = [0u64; 16];
        for (i, &q) in quotient.iter().enumerate() {
            let mut carry = 0u128;
            for (k, &d) in b.iter().enumerate() {
                let t = u128::from(check[i + k]) + u128::from(q) * u128::from(d) + carry;
                check[i + k] = t as u64;
                carry = t >> 64;
            }
            let mut k = i + b.len();
            while carry != 0 {
                let t = u128::from(check[k]) + carry;
                check[k] = t as u64;
                carry = t >> 64;
                k += 1;
            }
        }
        let mut carry = 0u128;
        for (k, limb) in check.iter_mut().enumerate() {
            let t = u128::from(*limb) + u128::from(remainder.get(k).copied().unwrap_or(0)) + carry;
            *limb = t as u64;
            carry = t >> 64;
        }
        assert_eq!(&check[..a.len()], a, "{a:x?} / {b:x?}");
        assert!(
            check[a.len()..].iter().all(|&limb| limb == 0),
            "{a:x?} / {b:x?}"
        );
        let divisor_limb = |k: usize| b.get(k).copied().unwrap_or(0);
        let below = (0..remainder.len())
            .rev()
            .find(|&k| remainder[k] != divisor_limb(k))
            .is_some_and(|k| remainder[k] < divisor_limb(k));
        assert!(below, "{a:x?} % {b:x?} is not below the divisor");
    }

    /// A three-limb dividend over a two-limb divisor, over divisors whose small top limb against a
    /// large second limb makes the estimate overshoot.
    ///
    /// The full sweeps against the model are the VM's and the runtime's, at every width they hold;
    /// this pins the shared body on its own, with no model to lean on.
    #[test]
    fn a_division_satisfies_its_identity() {
        // A splitmix64 stream, for operands with no dependency to draw them.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        for round in 0..20_000 {
            let b = [
                next(),
                if round % 2 == 0 {
                    1 + next() % 4
                } else {
                    next() | 1
                },
            ];
            assert_divides(&[next(), next(), next()], &b);
        }
    }

    /// The divisions that need the add-back, which random operands all but never reach.
    ///
    /// Hacker's Delight's `divmnu` cases for exactly this, its 32-bit digits scaled to 64: each
    /// leaves the estimate one too large after the second-limb test, so the subtraction borrows
    /// out of the window and the divisor is added back.
    #[test]
    fn the_add_back_repairs_an_estimate_one_too_large() {
        const TOP: u64 = 1 << 63;
        for (a, b) in [
            (&[3, 0, TOP, 0][..], &[1, 0, TOP >> 2][..]),
            (
                &[0, 0, TOP >> 32, (TOP >> 32) - 1][..],
                &[1, 0, TOP >> 32][..],
            ),
            (
                &[0, u64::MAX - 1, 0, TOP][..],
                &[u64::MAX >> 32, 0, TOP][..],
            ),
            (&[0, u64::MAX - 1, 0, TOP][..], &[u64::MAX, 0, TOP][..]),
            (
                &[0, (u64::MAX >> 32) - 1, TOP >> 32][..],
                &[u64::MAX >> 32, TOP >> 32][..],
            ),
        ] {
            assert_divides(a, b);
        }
    }

    /// Two-limb sums, differences, products and negations against `u128`, which wraps at the same
    /// width.
    #[test]
    fn two_limb_arithmetic_agrees_with_u128() {
        let split = |v: u128| [v as u64, (v >> 64) as u64];
        let join = |l: [u64; 2]| (u128::from(l[1]) << 64) | u128::from(l[0]);
        let values = [
            0,
            1,
            u128::from(u64::MAX),
            1 << 64,
            u128::MAX,
            1 << 127,
            0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        ];
        for &a in &values {
            for &b in &values {
                let mut res = [0u64; 2];
                add(&mut res, &split(a), &split(b));
                assert_eq!(join(res), a.wrapping_add(b), "{a:#x} + {b:#x}");
                sub(&mut res, &split(a), &split(b));
                assert_eq!(join(res), a.wrapping_sub(b), "{a:#x} - {b:#x}");
                mul(&mut res, &split(a), &split(b));
                assert_eq!(join(res), a.wrapping_mul(b), "{a:#x} * {b:#x}");
            }
            let mut value = split(a);
            negate(&mut value);
            assert_eq!(join(value), a.wrapping_neg(), "-{a:#x}");
        }
    }

    /// One limb, against `u128`.
    #[test]
    fn a_division_by_one_limb_agrees_with_u128() {
        for (a, d) in [
            (u128::MAX, 3u64),
            (0, 7),
            (1 << 64, u64::MAX),
            (12_345_678_901_234_567_890_123, 98_765),
        ] {
            let limbs = [a as u64, (a >> 64) as u64];
            let mut quotient = [0u64; 2];
            let rest = divide_by_limb(&mut quotient, &limbs, d);
            let q = (u128::from(quotient[1]) << 64) | u128::from(quotient[0]);
            assert_eq!(
                (q, u128::from(rest)),
                (a / u128::from(d), a % u128::from(d))
            );
        }
    }
}
