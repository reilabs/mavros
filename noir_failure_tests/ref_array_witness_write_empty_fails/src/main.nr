fn main(i: u32, expected: pub Field) {
    let d: [&mut Field; 0] = [];
    *d[i] = 9;
    assert_eq(i as Field, expected);
}
