fn main(x: u8) {
    let bytes: [u8; 33] = (x as Field).to_le_bytes();
    assert(bytes[32] == 0);
}
