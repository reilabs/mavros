//! A collection of miscellaneous utils for the compiler that don't necessarily have a good place.

use crate::compiler::ssa::hlssa::{MAX_SUPPORTED_SIGNED_BITS, MAX_SUPPORTED_UNSIGNED_BITS};

/// All-ones mask for the low `bits` bits. Total on the edges: `bits == 0` gives `0` and
/// `bits >= 128` gives all ones, so callers never need their own width guards.
pub fn bit_mask(bits: usize) -> u128 {
    if bits == 0 {
        0
    } else if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// Decode an `s`-bit two's-complement raw value (`raw < 2^s`) into the integer it represents.
pub fn decode_signed(s: usize, raw: u128) -> i128 {
    debug_assert!((1..=MAX_SUPPORTED_SIGNED_BITS).contains(&s));
    if (raw >> (s - 1)) & 1 == 1 {
        (raw as i128) - (1i128 << s)
    } else {
        raw as i128
    }
}

/// Whether `v` is representable in `s`-bit two's complement.
pub fn fits_signed(s: usize, v: i128) -> bool {
    v >= -(1i128 << (s - 1)) && v < (1i128 << (s - 1))
}

/// Encode `v` as an `s`-bit two's-complement raw value.
pub fn encode_signed(s: usize, v: i128) -> u128 {
    (v as u128) & bit_mask(s)
}

/// Widen a `from`-bit two's-complement raw value to `to` bits, replicating the sign bit.
///
/// Operates purely on the bit pattern, so it is correct whichever way the caller tags the result.
/// Narrowing (`to < from`) is not sign extension and is rejected rather than silently truncating.
pub fn sign_extend_bits(v: u128, from: usize, to: usize) -> u128 {
    debug_assert!(
        from > 0 && to >= from,
        "cannot sign-extend {from} bits to {to}"
    );
    let v = v & bit_mask(from);
    if (v >> (from - 1)) & 1 == 1 {
        v | (bit_mask(to) ^ bit_mask(from))
    } else {
        v
    }
}

/// Panic with the canonical ICE for a tuple surviving past the `ElideTuples` pass.
///
/// Everything downstream of `ElideTuples` operates on tuple-free IR; reaching a tuple opcode or
/// tuple type there is a compiler bug. Call this from the (unreachable) tuple arms of downstream
/// passes, analyses, and codegen.
#[track_caller]
pub fn ice_non_elided_tuple() -> ! {
    panic!("ICE: Tuple encountered after ElideTuples pass")
}

/// Panic if an `AssertConstant` marker survives its dedicated validation phase.
#[track_caller]
pub fn ice_unvalidated_assert_constant() -> ! {
    panic!("ICE: AssertConstant encountered after assert-constant validation")
}

pub fn spread_bits(v: u128, bits: usize) -> u128 {
    assert!(
        bits <= 64,
        "spread_bits only supports widths up to 64, got {bits}"
    );

    let mut x = v;
    x = (x | (x << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333u128;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555u128;
    x
}

pub fn unspread_bits(v: u128, bits: usize) -> (u128, u128) {
    assert!(
        bits <= MAX_SUPPORTED_UNSIGNED_BITS && bits % 2 == 0,
        "unspread_bits expects an even width up to {MAX_SUPPORTED_UNSIGNED_BITS}, got {bits}"
    );

    fn compact_bits(mut x: u128) -> u128 {
        x &= 0x5555_5555_5555_5555_5555_5555_5555_5555u128;
        x = (x | (x >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333u128;
        x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128;
        x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128;
        x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128;
        x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128;
        x = (x | (x >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFFu128;
        x
    }

    let even = compact_bits(v);
    let odd = compact_bits(v >> 1);
    (odd, even)
}

/// Extract odd/even bit streams from a 64-bit spread value.
pub fn unspread_u64(v: u64) -> (u32, u32) {
    let (odd, even) = unspread_bits(v as u128, 64);
    (odd as u32, even as u32)
}

/// Compute spread of a 32-bit value: interleave zero bits between each bit.
pub fn spread_u64(v: u32) -> u64 {
    spread_bits(v as u128, 32) as u64
}

/// Utilities only available in tests.
#[cfg(test)]
pub mod test {
    use mavros_artifacts::FieldConfig;

    use crate::compiler::{
        Field,
        ssa::{ValueId, hlssa::builder::HLEmitter},
    };

    /// Convert the provided `n` into a field value.
    pub fn fr(n: u64) -> Field {
        FieldConfig::bn254().constant(n)
    }

    /// `alloc` of a scalar `Ref<Field>` seeded with an inert default value (0).
    ///
    /// The constant is interned (never a block instruction), so the seed never shows up in
    /// `op_counts`; tests that care about the contents `store` to the cell afterward (the store
    /// overwrites the seed).
    pub fn falloc(e: &mut impl HLEmitter) -> ValueId {
        let init = e.field_const(fr(0));
        e.alloc(init)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_extend_bits_agrees_with_decode_encode() {
        // The authoritative definition of sign extension is "same integer, wider encoding".
        // Check the bit-twiddling against it exhaustively for every 4-bit and 8-bit value.
        for from in [4usize, 8] {
            for raw in 0..(1u128 << from) {
                for to in [from, from + 1, 16, 32, 64] {
                    assert_eq!(
                        sign_extend_bits(raw, from, to),
                        encode_signed(to, decode_signed(from, raw)),
                        "sign_extend_bits({raw}, {from}, {to})"
                    );
                }
            }
        }
    }

    #[test]
    fn sign_extend_bits_is_identity_at_equal_width() {
        for from in [1usize, 7, 8, 32] {
            for raw in [0u128, 1, bit_mask(from), bit_mask(from) >> 1] {
                assert_eq!(sign_extend_bits(raw, from, from), raw & bit_mask(from));
            }
        }
    }

    #[test]
    fn sign_extend_bits_masks_its_input() {
        // Bits above `from` are not part of the value and must not survive into the result.
        assert_eq!(sign_extend_bits(0xFF00 | 0x01, 8, 16), 0x0001);
        assert_eq!(sign_extend_bits(0xFF00 | 0x80, 8, 16), 0xFF80);
    }

    #[test]
    fn sign_extend_bits_handles_full_width() {
        // `to == 128` must not overflow the shift used to build the mask.
        assert_eq!(sign_extend_bits(1, 1, 128), u128::MAX);
        assert_eq!(sign_extend_bits(0, 1, 128), 0);
    }
}
