// A left shift whose operands are constant only *inside a specialized function*.
//
// That is the one place `Specializer`'s own constant folder ever sees a shift with
// two constant operands. SCS folds any shift whose operands are already constant
// long before the specializer runs, and the specializer is the only pass that
// propagates a call site's constants into a callee body -- so this shape, and
// nothing simpler, reaches the fold. Written any other way the test still passes
// and covers nothing.
//
// What is being pinned: `<<` **wraps**. `200u8 << 1` is 400, which keeps only its
// low eight bits, so the answer is 144. The fold used to be a bare `a_val << b_val`
// on the raw payloads, minting `Constant::U(8, 400)` -- a constant a whole bit wider
// than the type it claims. Under that fold `400 == 144` folds to false and the R1CS
// is statically rejected, so this test goes red rather than merely wrong.
//
// The `pow_32`-shaped loop is not incidental: it is what makes specializing this
// call profitable, and the fold is only reached for a candidate the cost gate
// accepts. Each iteration multiplies a *witness* by a bit of the exponent, which
// costs a constraint while the bit is an ordinary runtime value and folds away once
// it is constant. Measured at 67 raw constraints against 17 specialized; with the
// twelve calls below that is 600 saved against 88 of code, a ratio of 6.8 against
// the 5.0 the specializer requires. Shrink the loop or the call count and the
// candidate is rejected, the specialized body is discarded, and the fold stops
// being observable.
//
// The calls must also differ in their witness argument. Twelve identical calls are
// one call after CSE, which drops the savings back below the gate.

fn shl_pow(w: Field, v: u8, n: u8) -> u8 {
    let s = v << n;

    let b: [bool; 32] = (s as Field).to_le_bits();
    let mut r: Field = 1;
    for i in 1..33 {
        r *= r;
        r = (b[32 - i] as Field) * (r * w) + (1 - b[32 - i] as Field) * r;
    }
    assert(r != 0);

    s
}

fn main(w: Field) {
    let mut acc = w;
    for _ in 0..12 {
        acc = acc + w;
        assert(shl_pow(acc, 200, 1) == 144);
    }
}
