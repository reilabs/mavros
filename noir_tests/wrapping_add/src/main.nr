// Tests u32 wrapping addition pattern used by SHA-256.
// SHA-256 needs: ((a as u64 + b as u64) % (1 << 32)) as u32
// This exercises: u32->u64 cast, u64 addition, modulo, u64->u32 cast.
fn add_u32_wrapping(a: u32, b: u32) -> u32 {
    let wide_a = a as u64;
    let wide_b = b as u64;
    let sum = wide_a + wide_b;
    let modulus: u64 = 4294967296; // 1 << 32 = 2^32
    let wrapped = sum % modulus;
    wrapped as u32
}

fn main(a: u32, b: u32) {
    // Non-wrapping case: 100 + 200 = 300
    let r1 = add_u32_wrapping(100, 200);
    assert_eq(r1, 300);

    // Wrapping case: 0xFFFFFFFF + 1 should wrap to 0
    let r2 = add_u32_wrapping(0xFFFFFFFF, 1);
    assert_eq(r2, 0);

    // Wrapping case with inputs: 0xFFFFFFF0 + 0x20 = wraps to 0x10
    let r3 = add_u32_wrapping(a, b);
    assert_eq(r3, 16); // 0xFFFFFFF0 + 0x20 = 0x100000010, mod 2^32 = 0x10 = 16
}
