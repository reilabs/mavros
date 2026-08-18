// pop_back on a witness-length slice: the element comes from window position
// `start + log_len - 1` (a witness index, lowered via the eq/select scan) and the logical
// length shrinks by one while the physical slice keeps the popped slot as garbage.
fn main(w: Field, expected_elem: pub Field, expected_len: pub Field) {
    let mut s: [Field] = [10];
    if w == 1 {
        s = s.push_back(20);
    }
    let (s2, e) = s.pop_back();
    assert_eq(e, expected_elem);
    assert_eq(s2.len() as Field, expected_len);
    assert_eq(s2[0], 10);
}
