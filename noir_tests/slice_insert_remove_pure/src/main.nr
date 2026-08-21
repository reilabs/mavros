// insert/remove on a *pure-length* slice: purify leaves the ops untouched and the
// LowerSliceInsert/LowerSliceRemove scans run with pure bounds (selects fold when the index is
// pure). Covers the insert-at-len append edge and a round trip back to the original content.
fn main(x: Field, expected: pub Field) {
    let mut s: [Field] = [10, 20];
    s = s.insert(1, x);
    assert_eq(s.len() as Field, 3);
    assert_eq(s[0], 10);
    assert_eq(s[1], x);
    assert_eq(s[2], 20);
    s = s.insert(3, 40);
    assert_eq(s.len() as Field, 4);
    assert_eq(s[3], 40);
    let (s2, e) = s.remove(1);
    assert_eq(e, expected);
    assert_eq(s2.len() as Field, 3);
    assert_eq(s2[0], 10);
    assert_eq(s2[1], 20);
    assert_eq(s2[2], 40);
}
