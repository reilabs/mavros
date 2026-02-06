unconstrained fn unc_div(x: Field, y: Field) -> Field {
    x / y
}

fn main(x: pub Field, y: pub Field) {
    let result = unsafe { unc_div(x, y) };
    let result2 = unsafe { unc_div(10, 2) };
    assert_eq(y * result, x);
    assert_eq(2 * result2, 10);
}
