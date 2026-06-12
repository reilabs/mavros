// A witness value (laundered through an unconstrained hint) stored through a ref
// parameter: the callee's write must back-taint the caller's slot.
unconstrained fn identity_hint(x: Field) -> Field {
    x
}

fn make_witness_value(x: Field) -> Field {
    let y = unsafe { identity_hint(x) };
    assert_eq(y, x);
    y
}

fn store_through_ref(value: &mut Field, x: Field) {
    let y = make_witness_value(x);
    *value = y;
}

fn main(x: Field) {
    let mut value = 0;
    store_through_ref(&mut value, x);
    assert_eq(value, x);
}
