// A multiply by zero must not be reported as an overflow.
//
// The overflow check for a multiply is `MAX / rhs < lhs`, which has no answer at `rhs == 0`. The
// guarded lowering branches on that case; the unguarded one substitutes a safe divisor of `1`
// instead, on the argument that `MAX / 1 < lhs` is false for every `bits`-wide `lhs` and so
// already reports "no overflow". This is the test of that argument.
//
// Both operands come from the witness, so nothing folds and the substituted divisor is exercised
// at run time in every lane rather than at compile time.
//
// The signed half is the sharper case, and `s` is `INT_MIN` on purpose: its magnitude is
// `2^(bits-1)`, one _past_ `MAX_S`, so it is the only operand for which the bound has to be the
// sign-shifted `MAX_S + 1` rather than `MAX_S`. With `t == 0` the product's sign is `s`'s own,
// which is what shifts it.

unconstrained fn mul_u8(a: u8, b: u8) -> u8 {
    a * b
}

unconstrained fn mul_i8(a: i8, b: i8) -> i8 {
    a * b
}

fn main(x: u8, y: u8, s: i8, t: i8) {
    // Safety: the products are asserted below.
    let a = unsafe { mul_u8(x, y) };
    assert(a == 0);
    // Safety: as above; the zero operand on the left is the mirrored case.
    let b = unsafe { mul_u8(y, x) };
    assert(b == 0);

    // Safety: `INT_MIN * 0`, the corner the sign-shifted magnitude bound exists for.
    let c = unsafe { mul_i8(s, t) };
    assert(c == 0);
    // Safety: as above, mirrored.
    let d = unsafe { mul_i8(t, s) };
    assert(d == 0);
}
