// The guard-off world of `slice_insert_guarded`: the insert never runs, so its bounds assert
// must be suppressed and the speculative shift voted out by the merge selects. Same program as
// the sibling test; only the inputs differ.
fn main(w1: u8, w2: u8, expected0: pub u8, expected_len: pub u8) {
    let w1 = w1 as Field;
    let w2 = w2 as Field;
    let expected0 = expected0 as Field;
    let expected_len = expected_len as Field;

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
