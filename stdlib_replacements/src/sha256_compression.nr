// SHA-256 compression function replacement
// Replaces the stdlib sha256_compression lowlevel intrinsic.

use std::mavros::{spread, unspread};

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

fn wrapping_mul(a: u32, b: u32) -> u32 {
    let prod = a as u64 * b as u64;
    (prod & 0xffffffff) as u32
}

fn rotr(x: u32, right_divisor: u32, left_multiplier: u32) -> u32 {
    (x / right_divisor) | wrapping_mul(x, left_multiplier)
}

fn ch(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ (!e & g)
}

fn maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}

fn big_sigma0(a: u32) -> u32 {
    // rotr(2) ^ rotr(13) ^ rotr(22)
    rotr(a, 4, 1073741824) ^ rotr(a, 8192, 524288) ^ rotr(a, 4194304, 1024)
}

fn big_sigma1(e: u32) -> u32 {
    // rotr(6) ^ rotr(11) ^ rotr(25)
    rotr(e, 64, 67108864) ^ rotr(e, 2048, 2097152) ^ rotr(e, 33554432, 128)
}

fn small_sigma0(x: u32) -> u32 {
    // rotr(7) ^ rotr(18) ^ shr(3)
    rotr(x, 128, 33554432) ^ rotr(x, 262144, 16384) ^ (x / 8)
}

fn small_sigma1(x: u32) -> u32 {
    // rotr(17) ^ rotr(19) ^ shr(10)
    rotr(x, 131072, 32768) ^ rotr(x, 524288, 8192) ^ (x / 1024)
}

fn wrapping_add(a: u32, b: u32) -> u32 {
    let sum = a as u64 + b as u64;
    (sum & 0xffffffff) as u32
}

pub fn sha256_compression(msg_block: [u32; 16], state: [u32; 8]) -> [u32; 8] {
    // Message schedule
    let mut w: [u32; 64] = [0; 64];
    for i in 0..16 {
        w[i] = msg_block[i];
    }
    for i in 16..64 {
        w[i] = wrapping_add(
            wrapping_add(small_sigma1(w[i - 2]), w[i - 7]),
            wrapping_add(small_sigma0(w[i - 15]), w[i - 16]),
        );
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for i in 0..64 {
        let t1 = wrapping_add(
            wrapping_add(h, big_sigma1(e)),
            wrapping_add(ch(e, f, g), wrapping_add(K[i], w[i])),
        );
        let t2 = wrapping_add(big_sigma0(a), maj(a, b, c));

        h = g;
        g = f;
        f = e;
        e = wrapping_add(d, t1);
        d = c;
        c = b;
        b = a;
        a = wrapping_add(t1, t2);
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
