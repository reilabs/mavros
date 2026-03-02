// Tests bitwise AND (&) on u32 with multi-bit patterns (not just boolean).
// Needed by: SHA-256 choose function: Ch(e,f,g) = (e & f) ^ (~e & g)
fn main(a: u32, b: u32) {
    // Multi-bit AND (masking pattern)
    let r1 = a & b;
    assert_eq(r1, 0x00F0); // 0x0FF0 & 0x00FF = 0x00F0

    // AND as bit mask
    let x: u32 = 0xDEADBEEF;
    let mask: u32 = 0xFF000000;
    let high_byte = x & mask;
    assert_eq(high_byte, 0xDE000000);

    // AND with all ones (identity)
    let y: u32 = 0x12345678;
    let all_ones: u32 = 0xFFFFFFFF;
    assert_eq(y & all_ones, y);

    // AND with zero (annihilator)
    let zero: u32 = 0;
    assert_eq(y & zero, 0);
}
