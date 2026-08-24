// insert *guarded* by a witness condition on a witness-length slice: the insert's bounds
// assert is suppressed when the guard is off, and the speculative shift is voted out by the
// merge selects. Expected values are public inputs so the same circuit checks either world.
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
