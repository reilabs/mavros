use std::aes128::aes128_encrypt;

fn main(msg: [u8; 64], iv: [u8; 16], key: [u8; 16], expected: pub [u8; 80]) {
    assert_eq(aes128_encrypt(msg, iv, key), expected);
}
