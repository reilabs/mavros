// True-guarded dynamic index OOB
fn main(w: u8, i: u32, expected: pub u8) {
    let w = w as Field;
    let expected = expected as Field;

    let a: [Field; 3] = [1, 2, 3];
    let mut v = 7;
    if w == 1 {
        v = a[i];
    }
    assert_eq(v, expected);
}
