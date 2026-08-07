// Values at the sign boundary, where the unsigned and signed readings of the
// same bit pattern are furthest apart. `INT_MIN / -1` is the one combination
// that genuinely overflows, so it is never evaluated here - only the guard
// that would catch it gets emitted.
fn main(lo: i32, hi: i32, k: i32) {
    // lo = i32::MIN, hi = i32::MAX, k = 2.
    assert(lo < hi);
    assert(lo < 0);
    assert(hi > 0);
    assert(!(hi < lo));

    // Unsigned-reading order is the reverse of signed order for these two.
    assert(lo < k);
    assert(k < hi);

    assert_eq(lo / k, -1073741824);
    assert_eq(lo % k, 0);
    assert_eq(hi / k, 1073741823);
    assert_eq(hi % k, 1);

    // Same-width reinterpretation at the boundary. `cast_signed_unsigned`
    // already pins the general rule (`-1i32 as u32 == 4294967295`); this is the
    // extreme case of it. A round trip through `Field` is deliberately left
    // out: whether a negative widens by sign-extension or by raw zero-extension
    // is exactly the question the range-analysis rewrite has to settle, so
    // pinning an answer here before that work would be pinning a guess.
    assert_eq(lo as u32, 2147483648);
    assert_eq(hi as u32, 2147483647);

    // Adjacent to the boundary, avoiding overflow.
    assert_eq(lo + 1, -2147483647);
    assert_eq(hi - 1, 2147483646);

    // Narrower widths. i8::MIN has to be written as `-127 - 1`: Noir types the
    // literal before applying unary minus, so `-128` is `128` typed as i8,
    // which is `OverflowingConstant` (`hir_def/types.rs:318-329`). Both halves
    // of `-127 - 1` are representable and so is the result.
    let c: i8 = -127 - 1;
    let d: i8 = 127;
    assert(c < d);
    assert_eq(c / 2, -64);
    assert_eq(d / 2, 63);
    assert_eq(c as i32, -128);
}
