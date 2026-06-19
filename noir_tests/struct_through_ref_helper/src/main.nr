// realistic: a struct passed by `&mut` to a helper, then read back. The frontend
// decomposes the `&mut P` into one scalar Ref<Field> per field, which argument
// promotion then turns into by-value in/out params (here both fields are written,
// so `shift` gains two out-returns); mem2reg promotes the caller's struct alloc.
// Contrast argpromo_array_deferred, where a `&mut [Field; N]` keeps a single
// Ref<Array> and is deferred by C1.
struct P {
    x: Field,
    y: Field,
}

fn shift(p: &mut P, d: Field) {
    p.x = p.x + d;
    p.y = p.y - d;
}

fn main(x: Field, y: Field, d: Field, ex: pub Field, ey: pub Field) {
    let mut p = P { x, y };
    shift(&mut p, d);
    assert_eq(p.x, ex);
    assert_eq(p.y, ey);
}
