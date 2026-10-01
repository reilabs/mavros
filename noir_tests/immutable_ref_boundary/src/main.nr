struct Foo {
    x: Field,
    xs: [u8; 3],
}

unconstrained fn read_ref(r: &Field) -> Field {
    *r
}

unconstrained fn read_struct(r: &Foo) -> (Field, [u8; 3]) {
    (r.x, r.xs)
}

unconstrained fn mixed(a: &Field, offset: Field, b: &Field) -> Field {
    read_ref(a) + offset + read_ref(b)
}

unconstrained fn update(r: &mut Field) {
    *r = *r + 1;
}

unconstrained fn ordinary_calls(x: Field) -> Field {
    let mut value = x;
    update(&mut value);
    read_ref(&value)
}

fn main(x: Field, xs: [u8; 3]) {
    // Safety: each unconstrained result is checked against the constrained input.
    assert_eq(unsafe { read_ref(&x) }, x);
    let y = x + 1;
    assert_eq(unsafe { read_ref(&y) }, y);
    assert_eq(unsafe { mixed(&x, 2, &x) }, x + 2 + x);

    let foo = Foo { x, xs };
    let (field, array) = unsafe { read_struct(&foo) };
    assert_eq(field, x);
    assert_eq(array, xs);
    assert_eq(unsafe { ordinary_calls(x) }, x + 1);

    let sorted = xs.sort();
    assert_eq(sorted, [1, 2, 3]);
}
