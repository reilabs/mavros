// An unguarded pure `*` whose product leaves the width must be rejected.
//
// The multiply is the one of the three whose check is not a comparison against the wrapped result:
// there is no cheap wrap test for a product, so the condition is `MAX / rhs < lhs`, guarded against
// a zero divisor. The unguarded lowering writes that _flat_ -- it selects a safe divisor rather
// than branching on `rhs == 0` -- so this also pins that the zero case is handled without a
// diamond. The branching form is `LowerPureGuards::lower_unsigned_mul_guard`, which a guarded
// multiply still takes.
//
// Noir calls this "attempt to multiply with overflow".

fn mul_u8(a: u8, b: u8) -> u8 {
    a * b
}

fn main(w: Field) {
    // Keeps `main` from folding away entirely; the multiplication below is what is under test.
    assert(w != 0);
    // 200 * 2 is 400, above the 255 a `u8` holds. `144` is the wrapped answer.
    assert(mul_u8(200, 2) == 144);
}
