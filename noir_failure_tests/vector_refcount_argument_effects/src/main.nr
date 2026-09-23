fn checked_vector(x: Field) -> [Field] {
    assert_eq(x, 7);
    [x].as_vector()
}

fn main(x: Field) {
    // Returning a constant zero must not discard the argument's failing assertion.
    assert_eq(std::mem::vector_refcount(checked_vector(x)), 0);
}
