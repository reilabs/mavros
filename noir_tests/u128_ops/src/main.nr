// Tests u128 arithmetic and casting.
// Needed by: RSA/bignum packing (pack_be_bytes_into_u128s).
fn main(a: u64, b: u64) {
    // Cast u64 to u128
    let x = a as u128;
    let y = b as u128;

    // u128 addition
    let sum = x + y;
    // 0xFFFFFFFF + 0x00000001 = 0x100000000
    assert_eq(sum, 4294967296);

    // u128 multiplication (byte packing: result *= 256)
    let base: u128 = 256;
    let byte_val: u128 = 0xAB;
    let packed = base * byte_val;
    assert_eq(packed, 43776); // 256 * 171 = 43776

    // Multi-byte packing pattern (simulates pack_be_bytes_into_u128s)
    let b0: u128 = 0xDE;
    let b1: u128 = 0xAD;
    let mut result: u128 = 0;
    result = result * 256 + b0;
    result = result * 256 + b1;
    assert_eq(result, 0xDEAD);
}
