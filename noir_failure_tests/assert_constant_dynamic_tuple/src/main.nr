use std::assert_constant;

fn main(x: u8) {
    let x = x as Field;

    assert_constant((1, x));
}
