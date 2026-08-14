// Signed values crossing the constrained/unconstrained boundary in both
// directions, so the signed paths run in the unconstrained interpreter as
// well as in the circuit.
unconstrained fn abs_hint(x: i32) -> i32 {
    if x < 0 {
        -x
    } else {
        x
    }
}

unconstrained fn div_hint(x: i32, y: i32) -> i32 {
    x / y
}

fn main(x: i32, y: i32) {
    // Safety: the hint is checked against `-x` and `0` below.
    let a = unsafe { abs_hint(x) };
    assert(a >= 0);
    assert_eq(a, -x);
    assert_eq(a, 9);

    // Verify an unconstrained signed division against the constrained one.
    // Safety: the hint is checked against the constrained `x / y` below.
    let q = unsafe { div_hint(x, y) };
    assert_eq(q, x / y);
    assert_eq(q, -2);
}
