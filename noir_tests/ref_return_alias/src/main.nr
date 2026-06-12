// A callee returning an alias of its ref parameter: the witness written through the
// *returned* ref must be visible through the *original* binding.
fn id(p: &mut Field) -> &mut Field {
    p
}

fn main(x: Field) {
    let mut value = 0;
    let r = id(&mut value);
    *r = x;
    assert_eq(value, x);
}
