//! A collection of miscellaneous utils for the compiler that don't necessarily have a good place.

use crate::compiler::ssa::hlssa::MAX_SUPPORTED_UNSIGNED_BITS;

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
