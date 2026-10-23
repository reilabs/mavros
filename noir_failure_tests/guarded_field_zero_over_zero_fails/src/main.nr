// `0 / 0` on field elements in a taken branch must be rejected, as Noir rejects any field
// division by zero. The guarded lowering's `r * b == a` holds for every `r` at `0 / 0`, so only
// the zero check can reject it.
fn main(a: Field, b: Field, w: bool) -> pub Field {
    let mut r = 0;
    if w {
        r = a / b;
    }
    r
}
