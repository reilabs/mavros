// Tests shift left (<<) and shift right (>>) operators on u32.
// Needed by: SHA-256 rotations (right_rotate = (x >> n) | (x << (32-n))),
//            byte extraction from u32 words.
fn main(x: u32) {
    // Left shift by constant
    let shifted_left = x << 4;
    assert_eq(shifted_left, 672); // 42 << 4 = 672

    // Right shift by constant
    let shifted_right = x >> 4;
    assert_eq(shifted_right, 2); // 42 >> 4 = 2

    // Shift by constant (common in SHA-256)
    let a: u32 = 0xABCD1234;
    let b = a >> 8;
    assert_eq(b, 0x00ABCD12);

    let c = a << 8;
    assert_eq(c, 0xCD123400);
}
