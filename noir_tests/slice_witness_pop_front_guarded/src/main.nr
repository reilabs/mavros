// pop_front *guarded* by a witness condition on a witness-length slice that may be empty.
// With both guards off the speculative `start += 1, log_len - 1` runs against an empty
// window; the guarded rangecheck keeps the circuit satisfiable, mirroring the back-pop twin.
fn main(w1: u8, w2: u8, expected_len: pub u8) {
    let w1 = w1 as Field;
    let w2 = w2 as Field;
    let expected_len = expected_len as Field;

    let mut s: [Field] = [];
    if w1 == 1 {
        s = s.push_back(10);
    }
    if w2 == 1 {
        let (e, s2) = s.pop_front();
        s = s2;
        assert_eq(e, 10);
    }
    assert_eq(s.len() as Field, expected_len);
}
