use std::assert_constant;

fn requires_constant(value: Field) {
    assert_constant(value);
}

fn main(x: u8) {
    let x = x as Field;

    requires_constant(x);
}
