// remove at a witness index from a witness-length slice: the element is read at witness
// position `start + index` (eq/select scan) and the survivors shift left via a select scan
// against the witness index.
fn main(w: u32, expected_elem: pub Field, expected0: pub Field, expected_len: pub Field) {
    let mut s: [Field] = [10];
    if w == 1 {
        s = s.push_back(20);
    }
    let (s2, e) = s.remove(w);
    assert_eq(e, expected_elem);
    assert_eq(s2.len() as Field, expected_len);
    assert_eq(s2[0], expected0);
}
