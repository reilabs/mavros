// Field division by zero is an execution failure in Noir, but `div_field` answers 0 rather than
// trapping, so an unchecked one is a silent wrong answer. `LowerPureGuards` now asserts the
// divisor is nonzero for field division too.
//
// This pins the direction that costs something: ordinary field divisions must keep working, and
// must keep working across witness divisors, constant divisors, and inside witness-dependent
// control flow (where the guarded lowering, not this check, is responsible).
fn main(x: Field, y: Field, c: bool) {
    // Witness divisor.
    assert_eq(x / y, 4);

    // Constant divisor: the check folds away and must cost nothing.
    assert_eq(x / 2, 6);

    // Divisor that is a computed witness value rather than a parameter.
    let d = y + 1;
    assert_eq(x / d, 3);

    // Guarded: inside witness-dependent control flow the inactive branch must not fail even
    // though its divisor would be zero there.
    let z = if c { x / y } else { x };
    assert_eq(z, 4);

    // Chained, so the quotient of one division is the dividend of the next.
    assert_eq((x / y) / 2, 2);
}
