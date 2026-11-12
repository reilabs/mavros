// An unconstrained byte decomposition into more bytes than a field element holds fails the
// standard library's `static_assert` on the byte count, on every field.
unconstrained fn decompose(x: Field) -> [u8; 40] {
    x.to_le_bytes()
}

fn main(x: u8) -> pub u8 {
    // Safety: the decomposition fails before its bytes are used.
    let bytes = unsafe { decompose(x as Field) };
    bytes[0]
}
