// Slice bounds must use the runtime element count.
unconstrained fn read(values: [Field], index: u32) -> Field {
    values[index]
}

fn main(value: Field, index: u32) {
    let values = [value, value + 1].as_vector();
    // Safety: exercise an invalid unconstrained read; execution must trap.
    let result = unsafe { read(values, index) };
    assert_eq(result, 0);
}
