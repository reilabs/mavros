fn main(w: Field, i: u32, expected: pub Field) {
    let mut x = 3;
    let d = [&mut x];
    let mut v = 7;
    if w == 1 {
        v = *d[i];
    }
    assert_eq(v, expected);
}
