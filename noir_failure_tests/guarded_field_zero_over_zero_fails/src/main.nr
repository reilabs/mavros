// `0 / 0` on field elements in a taken branch must be rejected, as Noir rejects any field
// division by zero. The guarded lowering's `r * b == a` holds for every `r` at `0 / 0`, so only
// the zero check can reject it.
fn main(a: u8, b: u8, w: bool) -> pub bool {
    let a = a as Field;
    let b = b as Field;

    let mut r = 0;
    if w {
        r = a / b;
    }
    r == 0
}
