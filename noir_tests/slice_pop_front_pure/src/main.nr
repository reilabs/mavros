// pop_front on a *pure-length* slice: no witness control flow touches the length, so purify
// leaves the SlicePop untouched and `LowerSlicePop` rewrites the front pop late with a pure
// bound. The popped element is a witness value.
fn main(x: Field, expected: pub Field) {
    let mut s: [Field] = [10, 20];
    s = s.push_front(x);
    let (e, s2) = s.pop_front();
    assert_eq(e, expected);
    assert_eq(s2.len() as Field, 2);
    assert_eq(s2[0], 10);
    assert_eq(s2[1], 20);
    let (e2, s3) = s2.pop_front();
    assert_eq(e2, 10);
    assert_eq(s3.len() as Field, 1);
}
