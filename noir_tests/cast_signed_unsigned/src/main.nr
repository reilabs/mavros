fn main(
    a_u8: u8,          // 200
    b_i8_pos: i8,      // 42
    b_i8_neg: i8,      // -3
    c_u32: pub u32,    // 1000
    d_i32_neg: pub i32 // -1
) {
    // Unsigned widening: u8 -> u32
    let a_as_u32: u32 = a_u8 as u32;
    assert_eq(a_as_u32, 200);

    // Unsigned narrowing: u32 -> u8
    let c_as_u8: u8 = c_u32 as u8;
    assert_eq(c_as_u8, 232); // 1000 mod 256 = 232

    // Signed widening, positive value: i8 -> i32
    let pos_as_i32: i32 = b_i8_pos as i32;
    assert_eq(pos_as_i32, 42);

    // Signed widening, negative value: i8 -> i32 (sign extension)
    let neg_as_i32: i32 = b_i8_neg as i32;
    assert_eq(neg_as_i32, -3);

    // Unsigned to signed widening: u8 -> i32
    let u8_as_i32: i32 = a_u8 as i32;
    assert_eq(u8_as_i32, 200);

    // Signed to unsigned same width: i32 -> u32
    // -1 as i32 in twos complement = 2^32 - 1 = 4294967295
    let neg_as_u32: u32 = d_i32_neg as u32;
    assert_eq(neg_as_u32, 4294967295);

    // Unsigned to signed same width: u8 -> i8
    // 200 as u8 -> i8: 200 > 127 so it wraps to -56
    let u8_as_i8: i8 = a_u8 as i8;
    assert_eq(u8_as_i8, -56);

    // Signed narrowing: i32 -> i8
    let neg1_as_i8: i8 = d_i32_neg as i8;
    assert_eq(neg1_as_i8, -1); // -1 truncated to i8 is still -1

    // Signed to unsigned widening: i8 -> u32
    // Sign extends first: -3 as i8 -> -3 as i32 -> reinterpret as u32
    // 253 + (2^32 - 2^8) = 253 + 4294967040 = 4294967293
    let neg_i8_as_u32: u32 = b_i8_neg as u32;
    assert_eq(neg_i8_as_u32, 4294967293);
}
