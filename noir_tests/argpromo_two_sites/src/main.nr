// argument promotion: one helper called from two distinct sites, each passing a
// clean local. Promotion is all-or-nothing across call sites (C4), so both sites
// must agree; here both are clean, so the parameter is promoted.
fn inc(p: &mut Field) {
    *p = *p + 1;
}

fn main(x: Field, y: Field, ex: pub Field, ey: pub Field) {
    let mut a = x;
    inc(&mut a);
    let mut b = y;
    inc(&mut b);
    assert_eq(a, ex);
    assert_eq(b, ey);
}
