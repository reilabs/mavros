// Tests bitwise XOR (^) on u8 and u32.
// Needed by: SHA-256 sigma functions: sigma0(x) = ROTR2(x) ^ ROTR13(x) ^ ROTR22(x)
fn main(a: u8, b: u8, c: u32, d: u32) {
    // u8 XOR with non-overlapping bits
    let r1 = a ^ b;
    assert_eq(r1, 0xFF); // 0xAA ^ 0x55 = 0xFF

    // u32 XOR
    let r2 = c ^ d;
    assert_eq(r2, 0xFFFFFFFF); // 0xFF00FF00 ^ 0x00FF00FF = 0xFFFFFFFF

    // XOR self-inverse property
    let x: u32 = 0xDEADBEEF;
    let y: u32 = 0x12345678;
    let z = x ^ y;
    let w = z ^ y;
    assert_eq(w, x);
}
