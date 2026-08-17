// BLAKE2s replacement (RFC 7693): unkeyed, 32-byte digest.

use super::blake3::g;

global IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

global SIGMA: [[u32; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

fn round_fn(state: [u32; 16], m: [u32; 16], s: [u32; 16]) -> [u32; 16] {
    let mut v = state;
    // Mix the columns.
    let (a, b, c, d) = g(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
    v[0] = a;
    v[4] = b;
    v[8] = c;
    v[12] = d;
    let (a, b, c, d) = g(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
    v[1] = a;
    v[5] = b;
    v[9] = c;
    v[13] = d;
    let (a, b, c, d) = g(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
    v[2] = a;
    v[6] = b;
    v[10] = c;
    v[14] = d;
    let (a, b, c, d) = g(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
    v[3] = a;
    v[7] = b;
    v[11] = c;
    v[15] = d;
    // Mix the diagonals.
    let (a, b, c, d) = g(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
    v[0] = a;
    v[5] = b;
    v[10] = c;
    v[15] = d;
    let (a, b, c, d) = g(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
    v[1] = a;
    v[6] = b;
    v[11] = c;
    v[12] = d;
    let (a, b, c, d) = g(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
    v[2] = a;
    v[7] = b;
    v[8] = c;
    v[13] = d;
    let (a, b, c, d) = g(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    v[3] = a;
    v[4] = b;
    v[9] = c;
    v[14] = d;
    v
}

// `t` is the byte counter after this block; the high counter word stays IV[5] since N: u32.
fn compress(h: [u32; 8], block: [u32; 16], t: u32, last: bool) -> [u32; 8] {
    let v14 = if last { IV[6] ^ 0xFFFFFFFF } else { IV[6] };
    let mut v = [
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], IV[0], IV[1], IV[2], IV[3], IV[4] ^ t,
        IV[5], v14, IV[7],
    ];

    for r in 0..10 {
        v = round_fn(v, block, SIGMA[r]);
    }

    let mut out = h;
    for i in 0..8 {
        out[i] = h[i] ^ v[i] ^ v[i + 8];
    }
    out
}

pub fn blake2s<let N: u32>(input: [u8; N]) -> [u8; 32] {
    let mut padded = [0 as u8; ((N + 64) / 64) * 64];
    for i in 0..N {
        padded[i] = input[i];
    }

    let num_blocks = if N == 0 { 1 } else { (N + 63) / 64 };

    let mut h = IV;
    // Parameter block: digest length 32, no key, fanout 1, depth 1.
    h[0] = h[0] ^ 0x01010020;

    for b in 0..num_blocks {
        let mut block = [0 as u32; 16];
        for w in 0..16 {
            let base = b * 64 + w * 4;
            block[w] = padded[base] as u32
                + padded[base + 1] as u32 * 256
                + padded[base + 2] as u32 * 65536
                + padded[base + 3] as u32 * 16777216;
        }

        let last = b == num_blocks - 1;
        let t = if last { N } else { (b + 1) * 64 };
        h = compress(h, block, t, last);
    }

    let mut out = [0 as u8; 32];
    for i in 0..8 {
        let w = h[i];
        out[4 * i] = (w % 256) as u8;
        out[4 * i + 1] = ((w / 256) % 256) as u8;
        out[4 * i + 2] = ((w / 65536) % 256) as u8;
        out[4 * i + 3] = (w / 16777216) as u8;
    }
    out
}
