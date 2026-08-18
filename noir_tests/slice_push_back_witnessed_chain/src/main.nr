// Two chained witness-guarded back pushes. The second push targets an already-witness-length
// tuple, so its purify rewrite appends capacity at the physical end and then writes the value
// at the *witness* cursor `start + log_len`, which sits strictly inside the physical slice
// whenever the first guard was off (the appended copy becomes garbage). Expected values are
// public inputs so the same circuit checks different guard worlds.
fn main(w1: Field, w2: Field, expected1: pub Field, expected_len: pub Field) {
    let mut s: [Field] = [10];
    if w1 == 1 {
        s = s.push_back(20);
    }
    if w2 == 1 {
        s = s.push_back(30);
    }
    assert_eq(s.len() as Field, expected_len);
    assert_eq(s[0], 10);
    assert_eq(s[1], expected1);
}
