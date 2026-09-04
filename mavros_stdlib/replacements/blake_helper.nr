use crate::ops::WrappingAdd;

pub global IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

// Right rotations as div/mod by constant powers of two: mavros lowers `<<`
// with a rangecheck on the shifted value rather than wrapping, so
// `x << (32 - n)` on a full word would fail. rotr(x, n) is
// (x % 2^n) * 2^(32-n) + (x / 2^n), which never exceeds 32 bits.
fn rotr16(x: u32) -> u32 {
    (x % 65536) * 65536 + (x / 65536)
}

fn rotr12(x: u32) -> u32 {
    (x % 4096) * 1048576 + (x / 4096)
}

fn rotr8(x: u32) -> u32 {
    (x % 256) * 16777216 + (x / 256)
}

fn rotr7(x: u32) -> u32 {
    (x % 128) * 33554432 + (x / 128)
}

fn le_bytes_from_u32(w: u32) -> [u8; 4] {
    [(w % 256) as u8, ((w / 256) % 256) as u8, ((w / 65536) % 256) as u8, (w / 16777216) as u8]
}

// Serializes 8 words as a 32-byte little-endian digest.
pub fn le_bytes_from_words(words: [u32; 8]) -> [u8; 32] {
    let mut out = [0 as u8; 32];
    for i in 0..8 {
        let bytes = le_bytes_from_u32(words[i]);
        for j in 0..4 {
            out[4 * i + j] = bytes[j];
        }
    }
    out
}

pub fn g(a: u32, b: u32, c: u32, d: u32, mx: u32, my: u32) -> (u32, u32, u32, u32) {
    let a = a.wrapping_add(b).wrapping_add(mx);
    let d = rotr16(d ^ a);
    let c = c.wrapping_add(d);
    let b = rotr12(b ^ c);
    let a = a.wrapping_add(b).wrapping_add(my);
    let d = rotr8(d ^ a);
    let c = c.wrapping_add(d);
    let b = rotr7(b ^ c);
    (a, b, c, d)
}
