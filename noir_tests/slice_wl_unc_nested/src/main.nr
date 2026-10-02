unconstrained fn inner(xs: [Field]) -> [Field] {
    xs.push_back(4)
}

unconstrained fn outer(xs: [Field]) -> Field {
    let a = inner(xs.push_back(3));
    a.len() as Field + 10 * a[0] + 100 * a[1] + 1000 * a[2] + 10000 * a[3]
}

fn main(w: Field, expected: pub Field) {
    let mut s: [Field] = [1];
    if w == 1 {
        s = s.push_back(2);
    }
    let got = unsafe { outer(s) };
    assert_eq(got, expected);
}
