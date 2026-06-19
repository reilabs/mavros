// argument promotion: a scalar `&mut` parameter that is both read and written
// becomes a by-value in-param + an out-return. Follow-up mem2reg then promotes
// the caller's slot and the callee's materialized alloc.
fn add_to(p: &mut Field, delta: Field) {
    *p = *p + delta;
}

fn main(x: Field, delta: Field, expected: pub Field) {
    let mut v = x;
    add_to(&mut v, delta);
    assert_eq(v, expected);
}
