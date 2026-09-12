// Dynamic index OOB
fn main(i: u32, expected: pub Field) {
    let a: [Field; 3] = [1, 2, 3];
    assert_eq(a[i], expected);
}
