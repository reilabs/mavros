// `late_folded_witness_signed_divisor_dead_fails` for the other failure a signed division owes:
// `INT_MIN / -1`, whose quotient is one past the top of the type.
//
// `m` is `-1` only once `x` is promoted out of memory, so `a / m` is a dead signed division by a
// witness when the witness lowerings begin, and its lowering is the only thing that rejects this
// input. The cleanup has to keep it until then; if it does not, this program verifies.
fn main(a: i64, b: i64, c: u64) {
    let mut x = b;
    x += 0;
    let m = x - b - 1;
    let q = a / m;
    let _ = (q as u64) / c / c / c / c / c / c;
}
