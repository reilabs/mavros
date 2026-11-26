// Dynamic index on an empty array
fn main(i: u32, expected: pub u8) {
    let expected = expected as Field;

    let a: [Field; 0] = [];
    assert_eq(a[i], expected);
}
