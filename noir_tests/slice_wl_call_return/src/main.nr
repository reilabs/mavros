// A witness-length slice crossing call and return boundaries: the first call pushes under
// the caller's witness guard (pure slice in, wl decision inside the branch), the second call
// receives an already-witness-length slice as an argument and returns the grown tuple.
fn extend(s: [Field], v: Field) -> [Field] {
    s.push_back(v)
}

fn main(w: Field, e_last: pub Field, elen: pub Field) {
    let mut s: [Field] = [1];
    if w == 1 {
        s = extend(s, 2);
    }
    s = extend(s, 7);
    assert_eq(s.len() as Field, elen);
    assert_eq(s[0], 1);
    let last = s[s.len() - 1];
    assert_eq(last, e_last);
}
