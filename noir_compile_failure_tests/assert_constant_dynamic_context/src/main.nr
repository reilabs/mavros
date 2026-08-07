use std::assert_constant;

fn requires_constant(value: Field) {
    assert_constant(value);
}

fn main(x: Field) {
    requires_constant(x);
}
