// insert at a witness index into a witness-length slice: purify emits a physical-level
// SliceInsert at `start + index`, lowered into a grow-at-back plus a shift-right select scan
// against the witness index.
fn main(w: u32, expected0: pub u8, expected1: pub u8, expected2: pub u8, expected_len: pub u8) {
    let expected0 = expected0 as Field;
    let expected1 = expected1 as Field;
    let expected2 = expected2 as Field;
    let expected_len = expected_len as Field;

    let mut s: [Field] = [10];
    if w == 1 {
        s = s.push_back(20);
    }
    s = s.insert(w, 99);
    assert_eq(s.len() as Field, expected_len);
    assert_eq(s[0], expected0);
    assert_eq(s[1], expected1);
    assert_eq(s[2], expected2);
}
