fn main(x: Field) {
    let bits: [bool; 255] = x.to_le_bits();
    assert(!bits[254]);
}
