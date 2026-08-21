// remove *guarded* by a witness condition on a witness-length slice that may be empty. With
// both guards off the speculative remove runs against an empty window; the circuit stays
// satisfiable because the bounds assert is guarded and the shift is voted out by the merges.
fn main(w1: Field, w2: Field, expected_len: pub Field) {
    let mut s: [Field] = [];
    if w1 == 1 {
        s = s.push_back(10);
    }
    if w2 == 1 {
        let (s2, e) = s.remove(0);
        s = s2;
        assert_eq(e, 10);
    }
    assert_eq(s.len() as Field, expected_len);
}
