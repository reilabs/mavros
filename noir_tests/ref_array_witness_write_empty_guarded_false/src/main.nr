fn main(w: Field, i: u32, expected: pub Field) {
    let d: [&mut Field; 0] = [];
    let mut v = 7;
    if w == 1 {
        *d[i] = 9;
        v = 1;
    }
    assert_eq(v, expected);
}
