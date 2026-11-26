fn main(i: u32, expected: pub u8) {
    let expected = expected as Field;

    let d: [&mut Field; 0] = [];
    assert_eq(*d[i], expected);
}
