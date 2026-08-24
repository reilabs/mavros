// The guard-off world of `slice_insert_guarded`: the insert never runs, so its bounds assert
// must be suppressed and the speculative shift voted out by the merge selects. Same program as
// the sibling test; only the inputs differ.
fn main(w1: Field, w2: Field, expected0: pub Field, expected_len: pub Field) {
    let mut s: [Field] = [10];
    if w1 == 1 {
        s = s.push_back(20);
    }
    if w2 == 1 {
        s = s.insert(0, 5);
    }
    assert_eq(s.len() as Field, expected_len);
    assert_eq(s[0], expected0);
}
