// Runtime data keeps this read on the heap-array opcode, rather than BlobGet.
unconstrained fn read(value: Field, index: u32) -> Field {
    let values = [value, value + 1];
    values[index]
}

fn main(value: u8, index: u32) {
    let value = value as Field;

    // Safety: exercise an invalid unconstrained read; execution must trap.
    let result = unsafe { read(value, index) };
    assert_eq(result, 0);
}
