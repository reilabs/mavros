fn main(w: u8, i: u32, expected: pub u8) {
    let w = w as Field;
    let expected = expected as Field;

    let mut x = 3;
    let d = [&mut x];
    let mut v = 7;
    if w == 1 {
        v = *d[i];
    }
    assert_eq(v, expected);
}
