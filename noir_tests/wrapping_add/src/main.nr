// Tests u32 wrapping addition - the actual WrappingAdd trait method.
// Used by SHA-256 and SHA-1 round functions.
use std::ops::WrappingAdd;

fn main(a: u32, b: u32) {
    // Non-wrapping case: 100 + 200 = 300
    let r1: u32 = 100;
    let r2: u32 = 200;
    let sum = r1.wrapping_add(r2);
    assert_eq(sum, 300);

    // Wrapping case with witnesses
    let r3 = a.wrapping_add(b);
    assert_eq(r3, 16); // 0xFFFFFFF0 + 0x20 wraps to 0x10 = 16
}
