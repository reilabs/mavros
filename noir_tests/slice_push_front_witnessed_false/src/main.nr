// A front-push *guarded* by a witness condition. Untainting computes the push speculatively on
// both paths (the physical slice grows to the same pure length in both worlds) and the guard
// survives only in the merge selects over the slot contents and `log_len`. With the guard off,
// the pushed value is voted out slot-by-slot and the shorter window hides the garbage tail.
// Expected values are public inputs so the same program checks either world.
fn main(w: u8, expected0: pub u8, expected1: pub u8, expected_len: pub u8) {
    let w = w as Field;
    let expected0 = expected0 as Field;
    let expected1 = expected1 as Field;
    let expected_len = expected_len as Field;

    let mut s: [Field] = [10, 20];
    if w == 1 {
        s = s.push_front(5);
    }
    assert_eq(s.len() as Field, expected_len);
    assert_eq(s[0], expected0);
    assert_eq(s[1], expected1);
}
