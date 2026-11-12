// mem2reg + array SROA: a `let mut` array conditionally reassigned produces an
// aggregate-typed Alloc carrying an array phi. mem2reg promotes the alloc via an
// aggregate block-parameter phi, then array SROA peels it into per-cell scalars.
fn main(x: u8, sel: u8, e0: pub u8, e1: pub u8) {
    let x = x as Field;
    let sel = sel as Field;
    let e0 = e0 as Field;
    let e1 = e1 as Field;

    let mut a = [x, x + 1];
    if sel == 1 {
        a = [x + 10, x + 20];
    }
    assert_eq(a[0], e0);
    assert_eq(a[1], e1);
}
