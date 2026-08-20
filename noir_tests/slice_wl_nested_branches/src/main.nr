// Pushes inside nested witness branches: the inner push's effective guard is the conjunction
// w1 & w2, and the trailing push in the outer branch targets a tuple whose length already
// depends on the inner world. Inputs pick the world where the inner branch is off.
fn main(w1: Field, w2: Field, e_last: pub Field, elen: pub Field) {
    let mut s: [Field] = [1];
    if w1 == 1 {
        s = s.push_back(2);
        if w2 == 1 {
            s = s.push_back(3);
        }
        s = s.push_back(4);
    }
    assert_eq(s.len() as Field, elen);
    assert_eq(s[0], 1);
    let last = s[s.len() - 1];
    assert_eq(last, e_last);
}
