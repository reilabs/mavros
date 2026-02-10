pub struct A {
    pub a: Field,
    pub b: [Field; 2],
    pub c: B,
}

pub struct B {
    pub d: C,
    pub e: [Field; 2],
    pub f: Field,
}

pub struct C {
    pub g: Field,
}


fn main(x: [A; 3]) {
    assert_eq(x[0].a, 0);
    assert_eq(x[0].b[0], 1);
    assert_eq(x[0].b[1], 2);
    assert_eq(x[0].c.d.g, 3);
    assert_eq(x[0].c.e[0], 4);
    assert_eq(x[0].c.e[1], 5);
    assert_eq(x[0].c.f, 6);

    assert_eq(x[1].a, 7);
    assert_eq(x[1].b[0], 8);
    assert_eq(x[1].b[1], 9);
    assert_eq(x[1].c.d.g, 10);
    assert_eq(x[1].c.e[0], 11);
    assert_eq(x[1].c.e[1], 12);
    assert_eq(x[1].c.f, 13);

    assert_eq(x[2].a, 14);
    assert_eq(x[2].b[0], 15);
    assert_eq(x[2].b[1], 16);
    assert_eq(x[2].c.d.g, 17);
    assert_eq(x[2].c.e[0], 18);
    assert_eq(x[2].c.e[1], 19);
    assert_eq(x[2].c.f, 20);
}