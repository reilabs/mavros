// A division whose quotient is never read still has to be checked, because Noir treats a bad
// divisor as an execution failure whether or not anything reads the result. DCE therefore
// replaces such a division with its divisor check rather than deleting it outright.
//
// This test pins the *other* direction: a dead division with a perfectly good divisor must not
// acquire a check that can fail. Every division below is dead -- `println` bottoms out in an
// empty body, so the quotient has no consumer -- and every one of them is well defined, so the
// program must run clean. The failing direction is covered upstream by
// execution_failure/div_by_zero_{constants,numerator_witness,witness,modulo}.
fn main(x: u32, y: u32, s: i32, t: i32, f: Field, g: Field) {
    // Unsigned, witness divisor.
    let a: u32 = x / y;
    std::println(a);
    let b: u32 = x % y;
    std::println(b);

    // Signed, witness divisor: the check also covers INT_MIN / -1, so both operands matter.
    let c: i32 = s / t;
    std::println(c);
    let d: i32 = s % t;
    std::println(d);

    // Field, witness divisor.
    let e: Field = f / g;
    std::println(e);

    // Constant divisors: the check folds away entirely and must cost nothing.
    let h: u32 = x / 7;
    std::println(h);
    let i: i32 = s / 7;
    std::println(i);

    // A live division of the same shape, so the dead and live paths are exercised together.
    assert_eq(x / y, 4);
}
