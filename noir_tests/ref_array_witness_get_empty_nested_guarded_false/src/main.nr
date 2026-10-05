fn main(w: Field, i: u32, expected: pub Field) {
    let d: [([&mut Field; 2], Field); 0] = [];
    let mut v = 7;
    if w == 1 {
        let e = d[i];
        v = *e.0[0] + *e.0[1] + e.1;
    }
    assert_eq(v, expected);
}
