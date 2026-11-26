fn main(x: u8) {
    let bits: [bool; 255] = (x as Field).to_le_bits();
    assert(!bits[254]);
}
