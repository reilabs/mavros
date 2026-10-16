//! Wrapping arithmetic, shifts and orderings, and the spread and unspread bit rearrangements, on
//! buffers of little-endian `u64` limbs.
//!
//! Shared by the evaluators that compute wide integers in limbs, the VM and the WASM runtime's
//! helpers, and by the compiler, whose R1CS spread table uses [`spread_u32_to_u64`].
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
/// Schoolbook, straight into the result: the columns at or past `k`, and the carry out of column
/// `k-1`, are what the wrap discards, so they are never computed. A limb product plus two
/// limb-sized addends is at most `2^128 - 1`, so the accumulator cannot overflow.
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
/// Knuth's Algorithm D: each quotient limb is estimated from the top two dividend limbs over the
/// normalized divisor's top limb, at most two too large (Theorem B); the second-limb test brings
/// that to one, and the add-back repairs the last.
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

/// `a / b` and `a % b`, both read as unsigned, written into `quotient` and `remainder`, the four of
/// one length `k`.
///
/// Total: a zero divisor answers zero for both. `scratch` holds at least `2k + 1` limbs, whose
/// contents on entry do not matter; a divisor of one limb does not touch it.
pub fn udivrem(
    quotient: &mut [u64],
    remainder: &mut [u64],
    a: &[u64],
    b: &[u64],
    scratch: &mut [u64],
) {
    quotient.fill(0);
    remainder.fill(0);
    match significant_limbs(b) {
        0 => {}
        1 => remainder[0] = divide_by_limb(quotient, a, b[0]),
        n => {
            let (divisor, dividend) = scratch.split_at_mut(n);
            knuth_divide(quotient, remainder, a, &b[..n], divisor, dividend);
        }
    }
}

/// `a / b` and `a % b`, both read as two's complement at `bits`, written into `quotient` and
/// `remainder`, the four of one length `k` with every bit above `bits` clear.
///
/// Sign-magnitude around [`udivrem`]: the quotient takes the operands' xor and the remainder takes
/// the dividend's sign, which is truncation toward zero. Total: a zero divisor answers zero for
/// both, and `INT_MIN / -1` wraps to `INT_MIN` remainder zero. `scratch` holds at least `4k + 1`
/// limbs, whose contents on entry do not matter.
pub fn sdivrem(
    quotient: &mut [u64],
    remainder: &mut [u64],
    a: &[u64],
    b: &[u64],
    bits: usize,
    scratch: &mut [u64],
) {
    let k = a.len();
    let (a_negative, b_negative) = (is_negative(a, bits), is_negative(b, bits));
    let (a_store, rest) = scratch.split_at_mut(k);
    let (b_store, rest) = rest.split_at_mut(k);
    // Only a negative operand is copied, to be negated into its magnitude.
    let a_magnitude = if a_negative {
        negated_into(a_store, a, bits)
    } else {
        a
    };
    let b_magnitude = if b_negative {
        negated_into(b_store, b, bits)
    } else {
        b
    };

    udivrem(quotient, remainder, a_magnitude, b_magnitude, rest);

    if a_negative != b_negative {
        negate(quotient);
        mask_top(quotient, bits);
    }
    if a_negative {
        negate(remainder);
        mask_top(remainder, bits);
    }
}

/// `-value` at `bits`, written into `store` of the same length and returned.
fn negated_into<'a>(store: &'a mut [u64], value: &[u64], bits: usize) -> &'a [u64] {
    store.copy_from_slice(value);
    negate(store);
    mask_top(store, bits);
    store
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
    if bits - 64 * index > 64 {
        a[index]
    } else {
        a[index] & top_limb_mask(bits)
    }
}

/// The low `bits` bits of `a` spread into `res`, bit `i` to bit `2i`, with every other bit of `res`
/// cleared.
///
/// `a` holds at least the limbs `bits` reaches and `res` at least the limbs `2 * bits` does; the
/// two need not be the same length. Bits of `a` past `bits` are ignored, and limbs of `res` past
/// `2 * bits` are written as zero.
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
/// `a` holds at least the limbs `bits` reaches, and each stream at least the limbs its
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

// SHIFTS AND ORDERINGS
// ================================================================================================
//
// These read a value at a given width, not as its whole buffer: a shift reduces its amount modulo
// the width and wraps there, and a signed reading takes the width's top bit as the sign.
//
// A `bits`-wide value is `ceil(bits / 64)` limbs. Every bit above the width must be zero: a shift
// masks its value, but reads its amount's whole buffer, and an ordering compares both whole.

/// The bits the top limb of a `bits`-wide value holds, in `1..=64`.
#[must_use]
pub fn top_limb_bits(bits: usize) -> usize {
    debug_assert!(bits > 0, "a zero-width integer has no top limb");
    ((bits - 1) % 64) + 1
}

/// The mask of the bits the top limb of a `bits`-wide value holds.
#[must_use]
pub fn top_limb_mask(bits: usize) -> u64 {
    match top_limb_bits(bits) {
        64 => u64::MAX,
        held => (1 << held) - 1,
    }
}

/// Whether `value` is negative read as two's complement at `bits`: its bit `bits - 1`.
pub fn is_negative(value: &[u64], bits: usize) -> bool {
    (value[value.len() - 1] >> (top_limb_bits(bits) - 1)) & 1 == 1
}

/// Clear every bit of `value`'s top limb above `bits`.
pub fn mask_top(value: &mut [u64], bits: usize) {
    let top = value.len() - 1;
    value[top] &= top_limb_mask(bits);
}

/// The amount a shift by `b` applies to a `bits`-wide value: `b` read as unsigned, modulo `bits` (a
/// true modulo, which is a mask only where `bits` is a power of two). Zero at width zero.
pub fn shift_amount(b: &[u64], bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    if bits.is_power_of_two() {
        return b[0] as usize & (bits - 1);
    }
    let modulus = bits as u128;
    let mut rest = 0u128;
    for limb in b.iter().rev() {
        rest = ((rest << 64) | u128::from(*limb)) % modulus;
    }
    rest as usize
}

/// `a << amount` written into `res`, wrapping at `bits`, with `amount` already below `bits`.
pub fn shl(res: &mut [u64], a: &[u64], amount: usize, bits: usize) {
    let k = res.len();
    let limbs = amount / 64;
    let within = (amount % 64) as u32;
    for i in (0..k).rev() {
        let low = if i >= limbs { a[i - limbs] } else { 0 };
        let carried = if within > 0 && i > limbs {
            a[i - limbs - 1] >> (64 - within)
        } else {
            0
        };
        res[i] = (low << within) | carried;
    }
    mask_top(res, bits);
}

/// `a >> amount` written into `res`, zero-filling, with `amount` already below `bits`.
pub fn ushr(res: &mut [u64], a: &[u64], amount: usize, bits: usize) {
    shr(res, a, amount, bits, 0);
}

/// `a >> amount` written into `res`, sign-filling under a two's complement reading at `bits`, with
/// `amount` already below `bits`.
pub fn ashr(res: &mut [u64], a: &[u64], amount: usize, bits: usize) {
    let fill = if is_negative(a, bits) { u64::MAX } else { 0 };
    shr(res, a, amount, bits, fill);
}

/// The shared right shift, `fill` being the bits shifted in at the top.
fn shr(res: &mut [u64], a: &[u64], amount: usize, bits: usize, fill: u64) {
    let k = res.len();
    let limbs = amount / 64;
    let within = (amount % 64) as u32;
    let top_mask = top_limb_mask(bits);

    let limb = |i: usize| -> u64 {
        if i >= k {
            fill
        } else if i == k - 1 {
            (a[i] & top_mask) | (fill & !top_mask)
        } else {
            a[i]
        }
    };

    for i in 0..k {
        let high = limb(i + limbs);
        let carried = if within > 0 {
            limb(i + limbs + 1) << (64 - within)
        } else {
            0
        };
        res[i] = (high >> within) | carried;
    }
    mask_top(res, bits);
}

/// Whether `a < b`, reading both as unsigned, the two of one length.
pub fn ult(a: &[u64], b: &[u64]) -> bool {
    debug_assert_eq!(a.len(), b.len(), "an ordering across two lengths");
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// Whether `a < b`, reading both as two's complement at `bits`.
pub fn slt(a: &[u64], b: &[u64], bits: usize) -> bool {
    match (is_negative(a, bits), is_negative(b, bits)) {
        (true, false) => true,
        (false, true) => false,
        _ => ult(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread unspreads back to the bits it read, with nothing in the odd stream, at every width
    /// across three limbs, including the reads that end inside a limb's low half.
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

    /// A three-limb dividend over two-limb divisors whose small top limb and large second limb
    /// make the quotient estimate overshoot.
    ///
    /// The VM and the runtime sweep this body against the model at every width; this tests it
    /// directly, without the model.
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

    /// The divisions that need the add-back, which random operands almost never reach: Hacker's
    /// Delight's `divmnu` cases, scaled from 32-bit to 64-bit digits.
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
