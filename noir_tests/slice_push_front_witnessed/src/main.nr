// Front-push into a witness-length slice. The slice's length becomes witness-dependent via a
// conditional back-push, then a front-push prepends to it: under the `(physical, log_len, start)`
// purified representation, the physical slice grows at the front, the pushed value lands at
// `start`, and the logical length must track the witness.
fn main(w: u8, expected_len: pub u8) {
    let w = w as Field;
    let expected_len = expected_len as Field;

    let mut s: [Field] = [10];
    if w == 1 {
        s = s.push_back(20);
    }
    s = s.push_front(5);
    assert_eq(s.len() as Field, expected_len);
    assert_eq(s[0], 5);
    assert_eq(s[1], 10);
}
