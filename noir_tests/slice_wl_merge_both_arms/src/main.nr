// Differing-length merge where both arms are slice literals (an if-expression, no mutation
// and no pushes). The merged slice is then read through a witness index and, under a guard
// that is only live for the longer arm, at a pure index past the shorter arm's length.
fn main(w: Field, e_last: pub Field, elen: pub Field) {
    // `as_vector` on each arm: the arms are unified with each other before the `let` annotation
    // is consulted, so two array literals of different lengths would not type-check.
    let s: [Field] = if w == 1 { [1, 2].as_vector() } else { [5, 6, 7].as_vector() };
    assert_eq(s.len() as Field, elen);
    let last = s[s.len() - 1];
    assert_eq(last, e_last);
    if w == 0 {
        assert_eq(s[2], 7);
    }
}
