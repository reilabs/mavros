fn main(a: u8, b: pub u8, c: u32, d: pub u32) {
    // u8: both witness
    assert_eq(a & b, 0x89);
    assert_eq(a | b, 0xef);
    assert_eq(a ^ b, 0x66);

    // u8: witness op pure
    assert_eq(a & 0xff, a);
    assert_eq(a | 0x00, a);
    assert_eq(a ^ 0x00, a);
    assert_eq(a & 0x0f, 0x0b);

    // u32: both witness
    assert_eq(c & d, 0x12241668);
    assert_eq(c | d, 0xdebdfeff);
    assert_eq(c ^ d, 0xcc99e897);

    // u32: witness op pure
    assert_eq(c & 0xff, 0xef);
}
