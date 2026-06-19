fn store_through_nested(pp: &mut &mut Field, x: Field) {
    **pp = x;
}

fn main(x: Field) {
    let mut inner = 0;
    let mut p = &mut inner;
    store_through_nested(&mut p, x);
    assert_eq(inner, x);
}
