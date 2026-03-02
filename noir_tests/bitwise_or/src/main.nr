// Tests bitwise OR (|) on u8 and u32 with multi-bit patterns.
// Needed by: SHA-256 rotation synthesis: rotate = (x >> n) | (x << (32-n))
fn main(a: u8, b: u8, c: u32, d: u32) {
    // u8 bitwise OR
    let r1 = a | b;
    assert_eq(r1, 0xFF); // 0xF0 | 0x0F = 0xFF

    // u32 bitwise OR (simulates rotation combine)
    let r2 = c | d;
    assert_eq(r2, 0xFFFF); // 0xFF00 | 0x00FF = 0xFFFF

    // OR with overlapping bits
    let r3: u32 = 0xAA | 0x55;
    assert_eq(r3, 0xFF);

    // OR preserves set bits
    let r4: u32 = 0xDEAD0000 | 0x0000BEEF;
    assert_eq(r4, 0xDEADBEEF);
}
