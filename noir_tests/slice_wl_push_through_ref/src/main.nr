// Pushes routed through a &mut reference into a helper: the first call grows a pure-length
// slice, the second runs under a witness guard so the store through the reference writes a
// witness-length tuple back into the caller's slot.
fn grow(r: &mut [Field], v: Field) {
    *r = (*r).push_back(v);
}

fn main(w: Field, e_last: pub Field, elen: pub Field) {
    let mut s: [Field] = [10];
    grow(&mut s, 15);
    if w == 1 {
        grow(&mut s, 20);
    }
    assert_eq(s.len() as Field, elen);
    assert_eq(s[0], 10);
    assert_eq(s[1], 15);
    let last = s[s.len() - 1];
    assert_eq(last, e_last);
}
