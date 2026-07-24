fn main(message: pub str<11>) {
    let message_bytes: [u8; 11] = message.as_bytes();
    assert(message_bytes == [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]);

    let literal_bytes: [u8; 3] = "abc".as_bytes();
    assert(literal_bytes == [97, 98, 99]);
}
