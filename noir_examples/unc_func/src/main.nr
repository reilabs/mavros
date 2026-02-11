unconstrained fn unc_div(x: Field, y: Field) -> Field {
    x / y
}

fn main(x: pub Field, y: pub Field) {
    let result = unsafe { unc_div(x, y) };
    assert_eq(y * result, x);
}
