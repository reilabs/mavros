// A witnessed shift whose result only dies after taint inference must still check its amount.
//
// `x - s` is `s - s` only once `x` is promoted out of memory, after the one simplification that
// runs before taint inference, so `a << b` is live until then. The fold that leaves it dead comes
// after, and so does the dead-code elimination that deletes it. A shift with a witness operand gets
// no check from `LowerPureGuards`; its own lowering is what rejects an amount at the width, so the
// cleanup before that lowering has to keep it. If it does not, nothing checks `b`, and this
// program verifies with `b` at the width of a `u64`.
fn main(a: u64, b: u64) -> pub u64 {
    let s = a << b;
    let mut x = s;
    x - s
}
