fn main(w: Field, i: u32, expected: pub Field) {
    let d: [&mut Field; 0] = [];
    let mut v = 7;
    if w == 1 {
        v = *d[i];
    }
    assert_eq(v, expected);
}
