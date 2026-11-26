// argument promotion: a scalar `&mut` parameter that is only read (never stored
// through) becomes a by-value in-only param (no out-return), since nothing is
// written back.
fn get(p: &mut Field) -> Field {
    *p
}

fn main(x: u8, expected: pub u8) {
    let x = x as Field;
    let expected = expected as Field;

    let mut v = x + 1;
    let r = get(&mut v);
    assert_eq(r, expected);
}
