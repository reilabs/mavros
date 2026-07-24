fn main(bytes: pub [u8; 3]) {
    let string: str<3> = bytes.as_str_unchecked();
    assert(string.as_bytes() == bytes);

    let literal: str<2> = [104, 105].as_str_unchecked();
    assert(literal == "hi");
}
