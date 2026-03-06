// Tests LowLevel("sha256_compression") - the SHA-256 compression function.
// The passport program calls this 22 times via the sha256 library.
// This is a #[foreign(sha256_compression)] function in the Noir stdlib.
use std::hash::sha256_compression;

fn main(input: pub [u8; 4]) {
    // SHA-256 initial hash values (first 8 primes, fractional parts)
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Build a single 512-bit message block (16 x u32 words)
    // Pack the 4 input bytes into the first word, pad per SHA-256 spec
    let mut msg_block: [u32; 16] = [0; 16];
    msg_block[0] = (input[0] as u32) * 0x1000000
                  + (input[1] as u32) * 0x10000
                  + (input[2] as u32) * 0x100
                  + (input[3] as u32);
    // Append bit '1' after 4 bytes = set bit 31 of word 1
    msg_block[1] = 0x80000000;
    // Length in bits = 32, goes in last word
    msg_block[15] = 32;

    // Call the LowLevel sha256_compression
    state = sha256_compression(msg_block, state);

    // Verify first word of hash output for input [0x61, 0x62, 0x63, 0x64] = "abcd"
    // SHA-256("abcd") = 88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589
    assert(state[0] == 0x88d4266f);
}
