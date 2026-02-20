// SHA-256 compression function implementation following gnark-sha256.
// Reference: https://github.com/reilabs/gnark-sha256
//
// Follows gnark's direct 64-round Schedule/Permute structure.
// Uses division for right-shift and modular arithmetic for wrapping add,
// since Noir's >> and << operators are not available in the SSA backend.

global K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// Powers of two for u32: POW2[i] = 2^i for i in 0..32.
global POW2: [u32; 33] = [
    1, 2, 4, 8, 16, 32, 64, 128,
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
    0, // sentinel, unused
];

// Wrapping u32 addition: (a + b) mod 2^32.
fn wrapping_add(a: u32, b: u32) -> u32 {
    let sum: u64 = a as u64 + b as u64;
    let carry: u64 = sum / 4294967296;
    (sum - carry * 4294967296) as u32
}

// Right shift: x >> n, implemented as x / 2^n.
fn shr(x: u32, n: u32) -> u32 {
    x / POW2[n]
}

// Right rotation: (x >> n) | (x << (32-n)).
// Split x at position n and recombine.
fn rotr(x: u32, n: u32) -> u32 {
    let hi = x / POW2[n];
    let lo = x - hi * POW2[n];
    hi + lo * POW2[32 - n]
}

// Small sigma0: rotr(x,7) ^ rotr(x,18) ^ shr(x,3)
fn sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)
}

// Small sigma1: rotr(x,17) ^ rotr(x,19) ^ shr(x,10)
fn sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)
}

// Big Sigma0: rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)
fn big_sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}

// Big Sigma1: rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)
fn big_sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}

// Ch(e, f, g) = (e & f) ^ (~e & g) = g ^ (e & (f ^ g))
fn ch(e: u32, f: u32, g: u32) -> u32 {
    g ^ (e & (f ^ g))
}

// Maj(a, b, c) = (a & b) ^ (a & c) ^ (b & c)
fn maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}

// Message schedule: expand 16 input words to 64 words.
// Mirrors gnark-sha256's Schedule function.
fn schedule(input: [u32; 16]) -> [u32; 64] {
    let mut w: [u32; 64] = [0; 64];
    for i in 0..16 {
        w[i] = input[i];
    }
    for i in 16..64 {
        let s0 = sigma0(w[i - 15]);
        let s1 = sigma1(w[i - 2]);
        w[i] = wrapping_add(wrapping_add(wrapping_add(w[i - 16], s0), w[i - 7]), s1);
    }
    w
}

// Compression rounds: process 64 scheduled words against the state.
// Mirrors gnark-sha256's Permute function.
fn permute(state: [u32; 8], w: [u32; 64]) -> [u32; 8] {
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for i in 0..64 {
        let s1 = big_sigma1(e);
        let ch_val = ch(e, f, g);
        let temp1 = wrapping_add(
            wrapping_add(wrapping_add(wrapping_add(h, s1), ch_val), K[i]),
            w[i],
        );

        let s0 = big_sigma0(a);
        let maj_val = maj(a, b, c);
        let temp2 = wrapping_add(s0, maj_val);

        h = g;
        g = f;
        f = e;
        e = wrapping_add(d, temp1);
        d = c;
        c = b;
        b = a;
        a = wrapping_add(temp1, temp2);
    }

    [
        wrapping_add(state[0], a),
        wrapping_add(state[1], b),
        wrapping_add(state[2], c),
        wrapping_add(state[3], d),
        wrapping_add(state[4], e),
        wrapping_add(state[5], f),
        wrapping_add(state[6], g),
        wrapping_add(state[7], h),
    ]
}

// SHA-256 compression function.
// Takes a 16-word message block and an 8-word state, returns the updated state.
pub fn sha256_compression(input: [u32; 16], state: [u32; 8]) -> [u32; 8] {
    permute(state, schedule(input))
}
