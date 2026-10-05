fn main(i: u32, expected: pub Field) {
    let d: [&mut Field; 0] = [];
    assert_eq(*d[i], expected);
}
