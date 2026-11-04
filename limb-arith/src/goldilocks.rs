//! The Goldilocks prime field, `p = 2^64 - 2^32 + 1`, implemented on raw Montgomery-form `u64`
//! cells.
//!
//! This implementation backs `mavros_artifacts::Goldilocks`, which the compiler and the VM share,
//! and the WASM runtime's entry points, so that one implementation of reduction serves both.
//!
//! The Montgomery radix is `R = 2^64`, and a cell below `p` holds `x·R mod p` for the integer `x`
//! that it denotes. `p^-1 = 2^32 + 1 (mod 2^64)`, which is used by [`montgomery_reduce`].
//! `2^64 = 2^32 - 1 (mod p)` and `2^96 = -1 (mod p)`, so a 128-bit integer reduces with shifts,
//! one subtraction and one multiplication by `2^32 - 1` ([`reduce128`]).

/// The Goldilocks modulus, `2^64 - 2^32 + 1`.
pub const MODULUS: u64 = 0xffff_ffff_0000_0001;

/// `2^64 mod p = 2^32 - 1`, which is also `R mod p`, the Montgomery form of 1.
pub const EPSILON: u64 = 0xffff_ffff;

/// `p^-1 mod 2^64`: `p = 1 - 2^32 (mod 2^64)` and `(1 - 2^32)(1 + 2^32) = 1 - 2^64`.
const MODULUS_INVERSE: u64 = (1 << 32) + 1;

/// `R^2 mod p`, which takes a canonical integer into Montgomery form.
const R_SQUARED: u64 = ((EPSILON as u128 * EPSILON as u128) % MODULUS as u128) as u64;

/// `value·R^-1 mod p`, for `value < p·2^64`.
///
/// `m = lo·p^-1 (mod 2^64)` makes `m·p` agree with `value` in the low word, so `value - m·p` is
/// `(hi - (m·p)_hi)·2^64`. Both high words are below `p`, so the difference lies in `(-p, p)` and
/// one addition of `p` is enough to canonicalize.
#[inline]
pub const fn montgomery_reduce(value: u128) -> u64 {
    let lo = value as u64;
    let hi = (value >> 64) as u64;
    let m = lo.wrapping_mul(MODULUS_INVERSE);
    let mp_hi = ((m as u128 * MODULUS as u128) >> 64) as u64;
    let (difference, borrow) = hi.overflowing_sub(mp_hi);
    if borrow {
        difference.wrapping_add(MODULUS)
    } else {
        difference
    }
}

/// The Montgomery form of the canonical integer `value < p`.
#[inline]
pub const fn to_montgomery(value: u64) -> u64 {
    montgomery_reduce(value as u128 * R_SQUARED as u128)
}

/// The canonical integer the cell `cell` denotes.
#[inline]
pub const fn from_montgomery(cell: u64) -> u64 {
    montgomery_reduce(cell as u128)
}

/// `value mod p`, for any `u64`. A `u64` is below `2p`, so one subtraction suffices.
#[inline]
pub fn reduce64(value: u64) -> u64 {
    if value >= MODULUS {
        value - MODULUS
    } else {
        value
    }
}

/// `value mod p`, for any `u128`.
///
/// With `value = lo + 2^64·hi_lo + 2^96·hi_hi` (`hi_lo` and `hi_hi` the low and high 32 bits of
/// the high word), `value = lo - hi_hi + hi_lo·(2^32 - 1) (mod p)`.
#[inline]
pub fn reduce128(value: u128) -> u64 {
    let lo = value as u64;
    let hi = (value >> 64) as u64;
    let (hi_hi, hi_lo) = (hi >> 32, hi & EPSILON);

    // A borrow wrapped the difference by `+2^64`, which is `+(2^32 - 1)`, so we subtract that. The
    // borrow means `lo < hi_hi < 2^32`, so the wrapped difference is at least `2^64 - 2^32` and
    // this cannot borrow again.
    let (mut sum, borrow) = lo.overflowing_sub(hi_hi);
    if borrow {
        sum = sum.wrapping_sub(EPSILON);
    }

    // `hi_lo·(2^32 - 1) < 2^64`. A carry out of the sum wrapped it by `-2^64`, so add `2^32 - 1`
    // back; the wrapped sum is below `hi_lo·(2^32 - 1) <= 2^64 - 2^33 + 1`, so this cannot carry.
    let (sum, carry) = sum.overflowing_add(hi_lo * EPSILON);
    let sum = if carry { sum + EPSILON } else { sum };
    reduce64(sum)
}

/// `a + b` on cells below `p`.
///
/// A carried sum is `a + b - 2^64 < 2^64 - 2^33 + 2`, which stays below `2^64` once `2^32 - 1` is
/// added back.
#[inline]
pub fn add(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    let sum = if carry {
        sum.wrapping_add(EPSILON)
    } else {
        sum
    };
    reduce64(sum)
}

/// `a - b` on cells below `p`.
///
/// A borrowed difference is `a - b + 2^64`, so subtracting `2^64 - p = 2^32 - 1` gives `a - b + p`,
/// which is below `p` and cannot borrow again.
#[inline]
pub fn sub(a: u64, b: u64) -> u64 {
    let (difference, borrow) = a.overflowing_sub(b);
    if borrow {
        difference.wrapping_sub(EPSILON)
    } else {
        difference
    }
}

/// `a·b` on cells below `p`: `(a·R)(b·R)·R^-1 = (ab)·R`.
#[inline]
pub fn mul(a: u64, b: u64) -> u64 {
    montgomery_reduce(u128::from(a) * u128::from(b))
}

/// `-value` on a cell below `p`.
#[inline]
pub fn neg(value: u64) -> u64 {
    if value == 0 {
        0
    } else {
        MODULUS.wrapping_sub(value)
    }
}

/// `base` raised to `exp`, little-endian 64-bit limbs, on a cell below `p`.
pub fn pow(base: u64, exp: &[u64]) -> u64 {
    let mut result = EPSILON;
    for &limb in exp.iter().rev() {
        for bit in (0..64).rev() {
            result = mul(result, result);
            if (limb >> bit) & 1 == 1 {
                result = mul(result, base);
            }
        }
    }
    result
}

/// The multiplicative inverse of a cell below `p`, or `None` for zero: `value^(p - 2)`.
pub fn inverse(value: u64) -> Option<u64> {
    if value == 0 {
        None
    } else {
        Some(pow(value, &[MODULUS - 2]))
    }
}
