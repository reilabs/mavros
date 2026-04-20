use std::mavros::{unsafe_cast, spread, unspread};

fn main(x: u32, y: u32, cast_x: Field, cast_big: Field) {
    // --- unsafe_cast ---

    // Field -> u8: truncates to low 8 bits
    let x_u8: u8 = unsafe_cast(cast_x);
    assert_eq(x_u8, 200);

    // Field -> u16
    let x_u16: u16 = unsafe_cast(cast_x);
    assert_eq(x_u16, 200);

    // Field -> u32
    let big_u32: u32 = unsafe_cast(cast_big);
    assert_eq(big_u32, 1000);

    // Field -> u64
    let big_u64: u64 = unsafe_cast(cast_big);
    assert_eq(big_u64, 1000);

    // --- spread / unspread ---

    // spread<8> on x=5 (binary 101)
    // spread interleaves zero-bits: 101 -> 0_1_0_0_0_1 = 0x11 = 17
    let sx: u64 = spread::<8>(x);
    assert_eq(sx, 17);

    // unspread<8> should recover the original value and zero
    let (odd, even) = unspread::<8>(sx);
    assert_eq(odd, 0);
    assert_eq(even, x);

    // spread<8> on y=13 (binary 1101)
    // 1101 -> 0_1_0_1_0_0_0_1 = 0x51 = 81
    let sy: u64 = spread::<8>(y);
    assert_eq(sy, 81);

    let (odd2, even2) = unspread::<8>(sy);
    assert_eq(odd2, 0);
    assert_eq(even2, y);

    // Test spread then adding two spreads (the core use case for AND/XOR)
    let combined = sx + sy;
    let (result_and, result_xor) = unspread::<8>(combined);
    assert_eq(result_and, x & y);  // 5 & 13 = 5
    assert_eq(result_xor, x ^ y);  // 5 ^ 13 = 8
}
