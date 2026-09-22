// With four-byte elements this index wraps a wasm32 byte offset to zero.
unconstrained fn read(value: u32, index: u32) -> u32 {
    let values = [value, value];
    values[index]
}

fn main(value: u32, index: u32) {
    // Safety: an invalid read must trap even if its wrapped offset names a valid element.
    let result = unsafe { read(value, index) };
    assert_eq(result, value);
}
