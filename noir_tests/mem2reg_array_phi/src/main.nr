// mem2reg + array SROA: a `let mut` array conditionally reassigned produces an
// aggregate-typed Alloc carrying an array phi. mem2reg promotes the alloc via an
// aggregate block-parameter phi, then array SROA peels it into per-cell scalars.
fn main(x: Field, sel: Field, e0: pub Field, e1: pub Field) {
    let mut a = [x, x + 1];
    if sel == 1 {
        a = [x + 10, x + 20];
    }
    assert_eq(a[0], e0);
    assert_eq(a[1], e1);
}
