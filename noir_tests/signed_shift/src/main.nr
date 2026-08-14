// Shifts, signed and unsigned.
//
// Two limits in the witness shift lowering shape this test
// (`witness_bitwise.rs::lower_shift`):
//
//   * witness shift *amounts* are rejected outright (`:288`), and
//   * a witness *lhs* must be unsigned (`:291`).
//
// So a signed shift on a witness value has no lowering today at all. That gap
// is real and worth fixing, but it is not something this corpus entry can
// cover without turning into a permanent red row. What it does cover is the
// signed shift *opcodes* reaching constant folding, the lattice, and codegen,
// plus unsigned witness shifts as a control that the ordinary path still works.
fn main(a: u32, s: i32) {
    // Control: unsigned shifts on a witness value, constant amounts.
    assert_eq(a >> 2, 10);
    assert_eq(a << 1, 80);
    assert_eq(a >> 0, a);
    assert_eq(a << 0, a);

    // Signed shifts on non-witness values.
    let p: i32 = 40;
    assert_eq(p >> 2, 10);
    assert_eq(p << 1, 80);
    assert_eq(p >> 0, p);
    assert_eq(p << 0, p);

    // Negative lhs. A signed `>>` sign-fills, so the result stays negative
    // rather than becoming a huge positive number. This assertion is the whole
    // point of the arithmetic-shift fix: under the previous logical lowering
    // `neg >> 1` was 2147483644.
    let neg: i32 = -8;
    assert_eq(neg >> 1, -4);
    assert_eq(neg >> 2, -2);
    assert_eq(neg >> 3, -1);

    // Sign-fill saturates at -1 however far you shift, and 0 stays 0.
    assert_eq(neg >> 20, -1);
    let zero: i32 = 0;
    assert_eq(zero >> 5, 0);

    // -1 is all-ones, so it is a fixed point of the arithmetic shift.
    let neg_one: i32 = -1;
    assert_eq(neg_one >> 1, -1);

    // An odd negative rounds toward negative infinity, unlike signed division
    // which truncates toward zero: -7 >> 1 is -4, but -7 / 2 is -3.
    let odd: i32 = -7;
    assert_eq(odd >> 1, -4);
    assert_eq(odd / 2, -3);

    // `<<` is sign-agnostic on the bit pattern, so this one is exact.
    assert_eq(neg << 1, -16);

    // Narrower signed widths. The sign bit sits at `bits - 1`, not at 63, so
    // these catch a lowering that sign-extends from the wrong position.
    let e: i8 = -16;
    assert_eq(e << 1, -32);
    assert_eq(e >> 1, -8);
    assert_eq(e >> 7, -1);
    let f: i16 = 1024;
    assert_eq(f >> 3, 128);
    let g: i16 = -1024;
    assert_eq(g >> 3, -128);
    assert_eq(g >> 15, -1);
    let h: i64 = -1099511627776;
    assert_eq(h >> 10, -1073741824);

    // A signed value that came from a witness, shifted after being widened
    // through Field. Keeps a witness-derived signed value in play without
    // asking the witness shift lowering for something it cannot do.
    assert_eq(s * 4, 160);
}
