// mem2reg: a scalar `let mut` assigned different values on the two arms of an
// `if`, then read after the join. The local's Alloc + branch stores + load
// should be promoted to a block-parameter phi by mem2reg.
fn main(x: u8, sel: u8, expected: pub u8) {
    let x = x as Field;
    let sel = sel as Field;
    let expected = expected as Field;

    let mut v = x;
    if sel == 1 {
        v = v * 2;
    } else {
        v = v + 100;
    }
    assert_eq(v, expected);
}
