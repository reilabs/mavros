fn main(i: u32, expected: pub u8) {
    let expected = expected as Field;

    let mut x = 3;
    let d = [&mut x];
    assert_eq(*d[i], expected);
}
