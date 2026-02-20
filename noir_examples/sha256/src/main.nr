use std::hash::sha256_compression;

// SHA-256 initial hash values.
global H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn main(input: [u8; 5]) {
    // Manually pad the 5-byte message "hello" into a single 512-bit block.
    // Padding: message bytes | 0x80 | zeros | 64-bit big-endian length (40 bits)
    let mut block: [u32; 16] = [0; 16];
    // Pack bytes into big-endian u32 words using multiplication instead of <<
    block[0] = (input[0] as u32) * 16777216 + (input[1] as u32) * 65536 + (input[2] as u32) * 256 + (input[3] as u32);
    block[1] = (input[4] as u32) * 16777216 + 0x800000;
    // block[2..14] = 0 (already initialized)
    // Length in bits = 5 * 8 = 40, stored as big-endian u64 in last two words
    block[15] = 40;

    let result = sha256_compression(block, H);

    // SHA-256("hello") = 2cf24dba 5fb0a30e 26e83b2a c5b9e29e 1b161e5c 1fa7425e 73043362 938b9824
    assert(result[0] == 0x2cf24dba);
    assert(result[1] == 0x5fb0a30e);
    assert(result[2] == 0x26e83b2a);
    assert(result[3] == 0xc5b9e29e);
    assert(result[4] == 0x1b161e5c);
    assert(result[5] == 0x1fa7425e);
    assert(result[6] == 0x73043362);
    assert(result[7] == 0x938b9824);
}
