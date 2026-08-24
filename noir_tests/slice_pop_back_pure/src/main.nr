// pop_back on a *pure-length* slice: no witness control flow touches the length, so purify
// leaves the SlicePop untouched and `LowerSlicePop` rewrites it late into an assert + get +
// copy loop with a pure bound. The popped element is a witness value.
fn main(x: Field, expected: pub Field) {
    let mut s: [Field] = [10, 20];
    s = s.push_back(x);
    let (s2, e) = s.pop_back();
    assert_eq(e, expected);
    assert_eq(s2.len() as Field, 2);
    assert_eq(s2[0], 10);
    assert_eq(s2[1], 20);
    let (s3, e2) = s2.pop_back();
    assert_eq(e2, 20);
    assert_eq(s3.len() as Field, 1);
}
