// SHA-256 compression function implementation following the gnark-sha256 style.
// Reference: https://github.com/reilabs/gnark-sha256
//
// This replaces the #[foreign(sha256_compression)] function in Noir's stdlib
// with a pure Noir implementation suitable for R1CS constraint generation.
//
// Shift and modulo operators are not available, so we use:
//   right shift:  x / 2^n
//   left shift:   x * 2^n
//   rotation:     split at boundary, recombine
//   wrapping add: sum via u64, subtract carry * 2^32

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
global POW2: [u32; 32] = [
    1, 2, 4, 8, 16, 32, 64, 128,
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
];

// Wrapping u32 addition: (a + b) mod 2^32.
// Computes in u64 to avoid overflow, then extracts the low 32 bits
// by dividing out the carry.
fn wrapping_add(a: u32, b: u32) -> u32 {
    let sum: u64 = a as u64 + b as u64;
    let carry: u64 = sum / 4294967296;
    (sum - carry * 4294967296) as u32
}

// Right rotation of a 32-bit value by n positions.
// rotr(x, n) = (x >> n) | (x << (32 - n))
// Implemented by splitting x at position n and recombining.
fn rotr(x: u32, n: u32) -> u32 {
    let hi = x / POW2[n];           // top (32-n) bits
    let lo = x - hi * POW2[n];      // bottom n bits
    hi + lo * POW2[32 - n]          // recombine rotated
}

// Right shift of a 32-bit value by n positions.
fn shr(x: u32, n: u32) -> u32 {
    x / POW2[n]
}

// Small sigma0: used in message schedule
fn sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)
}

// Small sigma1: used in message schedule
fn sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)
}

// Big Sigma0: used in compression rounds
fn big_sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}

// Big Sigma1: used in compression rounds
fn big_sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}

// Choice: Ch(e, f, g) = (e AND f) XOR (NOT e AND g)
// Using algebraic identity to avoid NOT: g XOR (e AND (f XOR g))
fn ch(e: u32, f: u32, g: u32) -> u32 {
    g ^ (e & (f ^ g))
}

// Majority: Maj(a, b, c) = (a AND b) XOR (a AND c) XOR (b AND c)
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
