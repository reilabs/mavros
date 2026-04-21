use std::mavros::{unsafe_cast, spread, unspread};

fn main(x: u32, y: u32, a: u32, b: u32, cast_x: Field, cast_big: Field) {
    // --- unsafe_cast ---

    let x_u8: u8 = unsafe_cast(cast_x);
    assert_eq(x_u8, 200);
    let x_u16: u16 = unsafe_cast(cast_x);
    assert_eq(x_u16, 200);
    let big_u32: u32 = unsafe_cast(cast_big);
    assert_eq(big_u32, 1000);
    let big_u64: u64 = unsafe_cast(cast_big);
    assert_eq(big_u64, 1000);

    // --- spread<5> / unspread<5> ---

    let sx: u64 = spread::<5>(x);
    assert_eq(sx, 17); // spread(5) = 17
    let (odd, even) = unspread::<5>(sx);
    assert_eq(odd, 0);
    assert_eq(even, x);
    let sy: u64 = spread::<5>(y);
    assert_eq(sy, 81); // spread(13) = 81
    let combined5 = sx + sy;
    let (result_and5, result_xor5) = unspread::<5>(combined5);
    assert_eq(result_and5, x & y); // 5 & 13 = 5
    assert_eq(result_xor5, x ^ y); // 5 ^ 13 = 8

    // --- spread<7> / unspread<7> ---

    let sa: u64 = spread::<7>(a);
    assert_eq(sa, 5136); // spread(100) = 5136
    let (odd7, even7) = unspread::<7>(sa);
    assert_eq(odd7, 0);
    assert_eq(even7, a);
    let sb: u64 = spread::<7>(b);
    assert_eq(sb, 4177); // spread(77) = 4177
    let combined7 = sa + sb;
    let (result_and7, result_xor7) = unspread::<7>(combined7);
    assert_eq(result_and7, a & b); // 100 & 77 = 68
    assert_eq(result_xor7, a ^ b); // 100 ^ 77 = 41
}
