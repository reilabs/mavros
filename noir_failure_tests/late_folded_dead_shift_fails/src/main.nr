// A witnessed shift whose result dies only after taint inference must still check its amount.
//
// `x - s` becomes `s - s` only once `x` is promoted out of memory, after the one simplification
// that runs before taint inference, so `a << b` is live until then. The fold that makes it dead,
// and the dead-code elimination that deletes it, both come later. `LowerPureGuards` gives a shift
// with a witness operand no check, because its lowering rejects an amount at the width, so the
// cleanup before that lowering must keep it. Otherwise nothing checks `b`, and this program
// verifies with `b` equal to 64, the width of a `u64`.
fn main(a: u64, b: u64) -> pub u64 {
    let s = a << b;
    let mut x = s;
    x - s
}
