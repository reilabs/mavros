// pop_front on a witness-length slice: the element is read at the window start and the window
// advances (`start += 1, log_len -= 1`) with no physical shift. Afterward every read folds the
// now-nonzero `start` into its index, so the trailing `s2[0]` exercises a start-offset get.
fn main(w: Field, expected_elem: pub Field, expected_len: pub Field, expected0: pub Field) {
    let mut s: [Field] = [10];
    if w == 1 {
        s = s.push_back(20);
    }
    let (e, s2) = s.pop_front();
    assert_eq(e, expected_elem);
    assert_eq(s2.len() as Field, expected_len);
    assert_eq(s2[0], expected0);
}
