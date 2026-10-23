// The signed form of `late_folded_witness_divisor_dead_fails`.
//
// `LowerPureGuards` gives a signed division by a witness no check either, as its lowering rejects a
// zero divisor, so the cleanup after `z` is folded to zero must keep a dead signed division by a
// witness until that lowering has run. Otherwise the division is deleted with nothing checking `z`,
// and this program verifies.
//
// The levels peeled above it are unsigned. A signed division's check reads its dividend (for
// `INT_MIN / -1`), so a chain of signed divisions would keep every quotient below live, and `5 / z`
// would never be dead.
fn main(b: i64, c: u64) {
    let mut x = b;
    x += 0;
    let z = x - b;
    let q = 5 / z;
    let _ = (q as u64) / c / c / c / c / c / c;
}
