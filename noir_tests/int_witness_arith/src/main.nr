// Tests integer arithmetic on witness values across all widths (u8, u32, u64, u128).
// R1CS gen currently only handles integer arith on constants, not linear combinations.
// The passport uses all four widths: u8 (byte ops), u32 (SHA-256), u64 (SHA-512),
// u128 (RSA bignum).
fn main(a8: u8, b8: u8, a32: u32, b32: u32) {
    // u8 witness arithmetic
    let sum8 = a8 + b8;
    assert_eq(sum8, 15); // 10 + 5

    // u32 witness arithmetic
    let sum32 = a32 + b32;
    assert_eq(sum32, 300); // 100 + 200

    // u64 via cast from u32 witnesses
    let a64 = a32 as u64;
    let b64 = b32 as u64;
    let sum64 = a64 + b64;
    assert_eq(sum64, 300);

    // u128 via cast from u32 witnesses
    let a128 = a32 as u128;
    let b128 = b32 as u128;
    let sum128 = a128 + b128;
    assert_eq(sum128, 300);
}
