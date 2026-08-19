// Shifts, signed and unsigned, across every path the lowering has.
//
// There are three of them, and the point of this test is that all three agree:
//
//   * constant folding, for shifts whose operands are both known;
//   * `witness_bitwise.rs::lower_constant_amount_shift`, for an unsigned witness
//     value shifted by a constant, which folds `2^n` into a constant multiplier;
//   * `witness_bitwise.rs::lower_general_shift`, for a signed witness value or a
//     witness shift amount, which decomposes the amount and rebuilds `2^n` as a
//     product of per-bit factors.
//
// The third used to be a hard panic, which is why the signed cases below run on
// witness inputs rather than on locals: a local would be folded before the
// lowering ever saw it, and the interesting path would go untested.
//
// The last group pins the one semantic that is easy to get backwards: `<<`
// wraps, and only an out-of-range *amount* is a runtime error.
fn main(
    a: u32,
    n: u32,
    z: u32,
    w: u64,
    nw: u64,
    s: i32,
    t: i32,
    k: i32,
    zi: i32,
    e: i8,
    ne: i8,
    big: u32,
    ne4: i8,
) {
    // Unsigned witness value, constant amount. The pre-existing path.
    assert_eq(a >> 2, 10);
    assert_eq(a << 1, 80);
    assert_eq(a >> 0, a);
    assert_eq(a << 0, a);

    // Unsigned witness value, witness amount. `n` is 2 and `z` is 0, so these
    // also cover a shift that is a no-op without being a constant one.
    assert_eq(a >> n, 10);
    assert_eq(a << n, 160);
    assert_eq(a >> z, a);
    assert_eq(a << z, a);

    // A wider unsigned value, so the amount decomposition runs at six bits
    // rather than five. `w` is 2^60.
    assert_eq(w >> nw, 288230376151711744);
    assert_eq(w << nw, 4611686018427387904);

    // Signed shifts on non-witness values, folded before any lowering sees them.
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
    let c: i8 = -16;
    assert_eq(c << 1, -32);
    assert_eq(c >> 1, -8);
    assert_eq(c >> 7, -1);
    let f: i16 = 1024;
    assert_eq(f >> 3, 128);
    let g: i16 = -1024;
    assert_eq(g >> 3, -128);
    assert_eq(g >> 15, -1);
    let h: i64 = -1099511627776;
    assert_eq(h >> 10, -1073741824);

    // Signed witness value, constant amount. Non-negative first, where the sign
    // correction has to vanish rather than merely be small.
    assert_eq(s >> 2, 10);
    assert_eq(s << 1, 80);
    assert_eq(s >> 0, s);
    assert_eq(s << 0, s);

    // Signed witness value, negative, constant amount. `t` is -40, so `t >> 4`
    // is -3 and not -2: the shift floors where signed division truncates.
    assert_eq(t >> 1, -20);
    assert_eq(t >> 3, -5);
    assert_eq(t >> 4, -3);
    assert_eq(t / 16, -2);
    assert_eq(t << 1, -80);

    // Sign-fill saturates at the full width, which is the case where the
    // correction term is at its largest.
    assert_eq(t >> 31, -1);

    // A narrow signed witness value, to pin the sign bit position again on the
    // path that computes it rather than folds it. `e` is -16.
    assert_eq(e >> 1, -8);
    assert_eq(e << 1, -32);
    assert_eq(e >> 7, -1);

    // Signed witness value, witness amount: both halves of the old gap at once.
    // Noir requires the amount to have the lhs's exact type, so a signed value
    // is shifted by a signed amount -- which is the shape
    // `execution_success/inactive_signed_bitshift` has, and the reason the
    // amount's high-bit check has to treat a negative amount as out of range
    // rather than as a small one. `k` is 3, `zi` is 0, `ne` is 1.
    assert_eq(s >> k, 5);
    assert_eq(s << k, 320);
    assert_eq(t >> k, -5);
    assert_eq(t << k, -320);
    assert_eq(t >> zi, t);
    assert_eq(t << zi, t);
    assert_eq(e >> ne, -8);
    assert_eq(e << ne, -32);

    // A left shift WRAPS. Noir reports a runtime error only when the *amount*
    // reaches the width; bits pushed off the top are simply discarded. These
    // are therefore assertions about a value, not about a failure -- and they
    // are the cases that a lowering which rangechecks the product for overflow
    // gets wrong. `big` is 28 and `ne4` is 4.
    assert_eq(a << 28, 2147483648);
    assert_eq(a << big, 2147483648);
    assert_eq(w << 60, 0);
    assert_eq(e << 4, 0);
    assert_eq(e << ne4, 0);

    // Wrapping a signed value past its sign bit, where the result is negative
    // even though both operands are positive. `s` is 40, so `s << 26` is
    // 2684354560, which is -1610612736 read as i32.
    assert_eq(s << 26, -1610612736);
    assert_eq(e << 3, -128);
}
