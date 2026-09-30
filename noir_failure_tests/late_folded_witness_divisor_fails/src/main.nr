// A division by a witness whose divisor is only known to be zero after the guards are built must
// still be rejected.
//
// `x - b` is `b - b` only once `x` is promoted out of memory, which happens after the one
// algebraic simplification that runs before taint inference, so the subtraction is still a
// witness when `LowerPureGuards` meets `5 / z`. An unsigned division by a witness gets no zero
// check there, because its own lowering rejects a zero divisor. The simplification that folds
// `z` to zero runs after that, and has to leave it a witnessed zero rather than a bare one: a pure
// `5 / 0` would reach no witness lowering and no check at all.
fn main(b: u64) -> pub u64 {
    let mut x = b;
    x += 0;
    let z = x - b;
    5 / z
}
