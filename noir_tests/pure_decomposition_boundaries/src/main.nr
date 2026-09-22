fn main() {
    let _: [bool; 1] = (1 as Field).to_be_bits();
    let _: [bool; 1] = (1 as Field).to_le_bits();
    let _: [u8; 1] = (255 as Field).to_be_bytes();
    let _: [u8; 2] = (256 as Field).to_le_bytes();
    let _: [u8; 0] = (0 as Field).to_be_bytes();
    if false { let _: [bool; 1] = (2 as Field).to_be_bits(); let _: [u8; 1] = (256 as Field).to_le_bytes(); }
}
