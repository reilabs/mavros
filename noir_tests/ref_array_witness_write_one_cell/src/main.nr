fn main(i: u32, expected: pub Field) {
    let mut x = 3;
    let d = [&mut x];
    *d[i] = 9;
    assert_eq(x, expected);
}
