// The dead form of `late_folded_witness_divisor_fails`, which is the sharper of the two.
//
// Each dead-code elimination run before taint inference replaces the outermost of these divisions
// with its zero check but keeps the one below it, whose quotient that division read, so a run
// peels one level. Seven levels outlast those runs, leaving `5 / z` dead but present when the
// witness lowerings begin, with `z` still a witness and so given no check of its own there.
//
// Two things then keep it rejected. The simplification that folds `z` to zero has to leave it a
// witnessed zero, and the cleanup after it has to keep a dead unsigned division by a witness until
// its lowering has checked the divisor. If either stops holding, the division is deleted with
// nothing checking `z`, and this program verifies.
fn main(b: u64, c: u64) {
    let mut x = b;
    x += 0;
    let z = x - b;
    let _ = 5 / z / c / c / c / c / c / c;
}
