use std::assert_constant;

fn returns_unit() {}

fn main() {
    assert_constant(());
    assert_constant(returns_unit());
}
