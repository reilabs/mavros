// Front-push on a *pure-length* slice: no witness control flow touches the length, so purify
// leaves the SlicePush untouched and it flows to the backends as a first-class op (prepend at
// R1CS build time, memcpy-at-offset in witgen). The pushed element is a witness value, showing
// that element taint alone does not trigger the (physical, log_len, start) purification.
fn main(x: Field, expected: pub Field) {
    let mut s: [Field] = [10, 20];
    s = s.push_front(x);
    assert_eq(s.len() as Field, 3);
    assert_eq(s[0], expected);
    assert_eq(s[1], 10);
    assert_eq(s[2], 20);
}
