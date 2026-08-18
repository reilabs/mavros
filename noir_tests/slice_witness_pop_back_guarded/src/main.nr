// pop_back *guarded* by a witness condition on a witness-length slice that may be empty.
// With the pop guard off and the slice empty (w1 = 0, w2 = 0), the speculative `log_len - 1`
// underflows; the circuit stays satisfiable because the subtraction's rangecheck is guarded
// and its result is zeroed under a false guard. Expected values are public inputs so the same
// circuit checks the different guard worlds.
fn main(w1: Field, w2: Field, expected_len: pub Field) {
    let mut s: [Field] = [];
    if w1 == 1 {
        s = s.push_back(10);
    }
    if w2 == 1 {
        let (s2, e) = s.pop_back();
        s = s2;
        assert_eq(e, 10);
    }
    assert_eq(s.len() as Field, expected_len);
}
