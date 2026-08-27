// Widening a signed integer into a u128, which is the case the sign-extension
// lowering used to reject outright: it bounded the *target* width by
// `MAX_SUPPORTED_SIGNED_BITS`, when that cap describes how wide a signed
// *reading* goes. The target is only a wider integer to deposit the result in.
//
// `execution_success/cast_to_u128` covers the same ground upstream; this pins
// it locally, and adds the widths and the positive case that one does not.
// Every input is a witness on purpose - a folded cast takes `eval_sext`, which
// handled 128 bits all along, so only a witness reaches the lowering at all.
fn main(a: i8, b: i32, c: i16, d: i64) {
    // a = -1. All ones at every width, so the extension fills the whole target.
    assert_eq(a as u128, 340282366920938463463374607431768211455);

    // b = i32::MIN, so the answer is `2^128 - 2^31`: the raw pattern `2^31`
    // plus the extension term `2^128 - 2^32`. Exactly one sign bit is set and
    // nothing else, which is the case where that term has to be right rather
    // than merely large.
    assert_eq(b as u128, 340282366920938463463374607429620727808);

    // c = 42, non-negative, so no extension is added at all. This is the arm
    // the range domain can discharge, and it must still land on the plain
    // magnitude.
    assert_eq(c as u128, 42);

    // d = -3 from the widest signed source there is.
    assert_eq(d as u128, 340282366920938463463374607431768211453);

    // The intermediate widths still work, so the target width is genuinely a
    // parameter rather than a special case for 128.
    assert_eq(a as u64, 18446744073709551615);
    assert_eq(a as u32, 4294967295);
}
