// insert *guarded* by a witness condition on a witness-length slice: the insert's bounds
// assert is suppressed when the guard is off, and the speculative shift is voted out by the
// merge selects. Expected values are public inputs so the same circuit checks either world.
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
