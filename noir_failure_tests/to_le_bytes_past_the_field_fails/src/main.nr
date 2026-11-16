fn main(x: Field) {
    let bytes: [u8; 33] = x.to_le_bytes();
    assert(bytes[32] == 0);
}
