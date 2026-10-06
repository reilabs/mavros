fn main(i: u32, expected: pub Field) {
    let mut x = 3;
    let d = [&mut x];
    assert_eq(*d[i], expected);
}
