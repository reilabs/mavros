// A front-push *guarded* by a witness condition. Untainting computes the push speculatively on
// both paths (the physical slice grows to the same pure length in both worlds) and the guard
// survives only in the merge selects over the slot contents and `log_len`. With the guard off,
// the pushed value is voted out slot-by-slot and the shorter window hides the garbage tail.
// Expected values are public inputs so the same program checks either world.
fn main(w: Field, expected0: pub Field, expected1: pub Field, expected_len: pub Field) {
    let mut s: [Field] = [10, 20];
    if w == 1 {
        s = s.push_front(5);
    }
    assert_eq(s.len() as Field, expected_len);
    assert_eq(s[0], expected0);
    assert_eq(s[1], expected1);
}
