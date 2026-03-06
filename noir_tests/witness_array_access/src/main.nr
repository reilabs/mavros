// Tests array access with witness-derived index.
// The passport's SHA-256 library builds message blocks using indices
// computed from witness values (e.g., msg_byte_ptr based on message length).
// This exercises witness-tainted array get/set in the untaint_control_flow pass.
fn main(data: [u8; 16], idx: u32) {
    // Read from array at witness-derived index
    let val = data[idx];
    assert(val != 0);

    // Write to array at witness-derived index
    let mut output: [u8; 16] = [0; 16];
    output[idx] = val;
    assert_eq(output[idx], data[idx]);
}
