// Three chained pushes (back, front, back) under one witness guard: each push targets the
// tuple produced by the previous one, so cursors compound (front push renumbers indices, the
// second back push writes at start + log_len after two bumps). The final element is read back
// through a witness index.
fn main(w: Field, e0: pub Field, e_last: pub Field, elen: pub Field) {
    let mut s: [Field] = [10];
    if w == 1 {
        s = s.push_back(20);
        s = s.push_front(5);
        s = s.push_back(30);
    }
    assert_eq(s.len() as Field, elen);
    assert_eq(s[0], e0);
    let last = s[s.len() - 1];
    assert_eq(last, e_last);
}
