// argument promotion: two distinct (non-aliasing) scalar `&mut` parameters both
// promote to by-value in + out-return; the out-returns must align by parameter
// order so the writes land back on the right caller slots.
fn swap_add(a: &mut Field, b: &mut Field) {
    let t = *a;
    *a = *b + 1;
    *b = t + 2;
}

fn main(x: u8, y: u8, ea: pub u8, eb: pub u8) {
    let x = x as Field;
    let y = y as Field;
    let ea = ea as Field;
    let eb = eb as Field;

    let mut a = x;
    let mut b = y;
    swap_add(&mut a, &mut b);
    assert_eq(a, ea);
    assert_eq(b, eb);
}
