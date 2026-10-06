// The signed form of `late_folded_witness_divisor_dead_fails`.
//
// A signed division by a witness gets no check from `LowerPureGuards` either, as its lowering
// rejects a zero divisor itself, so the cleanup after the simplification that folds `z` to zero
// has to keep a dead signed division by a witness until that lowering has run. If it stops doing
// so, the division is deleted with nothing checking `z`, and this program verifies.
//
// The levels peeled above it are unsigned. A signed division's own check reads its dividend, for
// `INT_MIN / -1`, so a chain of signed ones would keep every quotient below it live, and `5 / z`
// would never be dead.
fn main(b: i64, c: u64) {
    let mut x = b;
    x += 0;
    let z = x - b;
    let q = 5 / z;
    let _ = (q as u64) / c / c / c / c / c / c;
}
