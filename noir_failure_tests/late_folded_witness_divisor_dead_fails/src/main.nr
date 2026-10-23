// The dead form of `late_folded_witness_divisor_fails`, and the sharper of the two.
//
// Each dead-code elimination run before taint inference replaces the outermost division with its
// zero check but keeps the one below, whose quotient it read, so each run peels one level. Seven
// levels outlast those runs, so `5 / z` is dead but present when the witness lowerings begin, with
// `z` still a witness and so given no separate check.
//
// Two things keep it rejected: the simplification that folds `z` to zero must leave a witnessed
// zero, and the cleanup after it must keep a dead unsigned division by a witness until its lowering
// has checked the divisor. If either fails, the division is deleted with nothing checking `z`, and
// this program verifies.
fn main(b: u64, c: u64) {
    let mut x = b;
    x += 0;
    let z = x - b;
    let _ = 5 / z / c / c / c / c / c / c;
}
