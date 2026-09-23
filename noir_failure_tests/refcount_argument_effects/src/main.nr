fn checked_array(x: Field) -> [Field; 1] {
    assert_eq(x, 7);
    [x]
}

fn main(x: Field) {
    // Returning a constant zero must not discard the argument's failing assertion.
    assert_eq(std::mem::array_refcount(checked_array(x)), 0);
}
