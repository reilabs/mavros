// A division by a witness whose divisor is known to be zero only after the guards are built must
// still be rejected.
//
// `x - b` becomes `b - b` only once `x` is promoted out of memory, after the one simplification
// that runs before taint inference, so the subtraction is still a witness when `LowerPureGuards`
// meets `5 / z`. It gives an unsigned division by a witness no zero check, because the lowering
// rejects a zero divisor. The simplification that later folds `z` to zero must leave a witnessed
// zero: a pure `5 / 0` would reach no witness lowering and no check at all.
fn main(b: u64) -> pub u64 {
    let mut x = b;
    x += 0;
    let z = x - b;
    5 / z
}
