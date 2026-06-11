// BLAKE3 replacement, naive bitwise implementation.
// Replaces the stdlib blake3 lowlevel intrinsic.
//
// Only handles single-chunk inputs (N <= 1024 bytes, i.e. up to 16 blocks),
// which is sufficient because the stdlib wrapper static_asserts N <= 1024.
// For a single chunk no merkle tree is built: the chunk's last block is
// compressed with the ROOT flag and the first 8 output words are the digest.

use crate::ops::WrappingAdd;

global IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

global MSG_PERMUTATION: [u32; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

global CHUNK_START: u32 = 1;
global CHUNK_END: u32 = 2;
global ROOT: u32 = 8;

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

fn g(a: u32, b: u32, c: u32, d: u32, mx: u32, my: u32) -> (u32, u32, u32, u32) {
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

fn round_fn(state: [u32; 16], m: [u32; 16]) -> [u32; 16] {
    let mut v = state;
    // Mix the columns.
    let (a, b, c, d) = g(v[0], v[4], v[8], v[12], m[0], m[1]);
    v[0] = a;
    v[4] = b;
    v[8] = c;
    v[12] = d;
    let (a, b, c, d) = g(v[1], v[5], v[9], v[13], m[2], m[3]);
    v[1] = a;
    v[5] = b;
    v[9] = c;
    v[13] = d;
    let (a, b, c, d) = g(v[2], v[6], v[10], v[14], m[4], m[5]);
    v[2] = a;
    v[6] = b;
    v[10] = c;
    v[14] = d;
    let (a, b, c, d) = g(v[3], v[7], v[11], v[15], m[6], m[7]);
    v[3] = a;
    v[7] = b;
    v[11] = c;
    v[15] = d;
    // Mix the diagonals.
    let (a, b, c, d) = g(v[0], v[5], v[10], v[15], m[8], m[9]);
    v[0] = a;
    v[5] = b;
    v[10] = c;
    v[15] = d;
    let (a, b, c, d) = g(v[1], v[6], v[11], v[12], m[10], m[11]);
    v[1] = a;
    v[6] = b;
    v[11] = c;
    v[12] = d;
    let (a, b, c, d) = g(v[2], v[7], v[8], v[13], m[12], m[13]);
    v[2] = a;
    v[7] = b;
    v[8] = c;
    v[13] = d;
    let (a, b, c, d) = g(v[3], v[4], v[9], v[14], m[14], m[15]);
    v[3] = a;
    v[4] = b;
    v[9] = c;
    v[14] = d;
    v
}

fn permute(m: [u32; 16]) -> [u32; 16] {
    let mut out = [0 as u32; 16];
    for i in 0..16 {
        out[i] = m[MSG_PERMUTATION[i]];
    }
    out
}

// Both counter words are always zero: the only chunk is chunk 0, and the
// first 32 bytes of root output are output block 0.
fn compress(cv: [u32; 8], block: [u32; 16], block_len: u32, flags: u32) -> [u32; 16] {
    let mut v = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], IV[0], IV[1], IV[2], IV[3], 0, 0,
        block_len, flags,
    ];
    let mut m = block;
    for _r in 0..7 {
        v = round_fn(v, m);
        m = permute(m);
    }
    let mut out = v;
    for i in 0..8 {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ cv[i];
    }
    out
}

pub fn blake3<let N: u32>(input: [u8; N]) -> [u8; 32] {
    // Zero-pad to the maximum single-chunk size so block-word extraction
    // below needs no bounds conditionals. All indices are compile-time
    // constants after loop unrolling, so the padding folds away.
    let mut padded = [0 as u8; 1024];
    for i in 0..N {
        padded[i] = input[i];
    }

    let num_blocks = if N == 0 { 1 } else { (N + 63) / 64 };

    let mut cv = IV;
    let mut root_state = [0 as u32; 16];
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
        let start_flag = if b == 0 { CHUNK_START } else { 0 };
        let end_flag = if last { CHUNK_END | ROOT } else { 0 };
        let block_len = if last { N - b * 64 } else { 64 };

        root_state = compress(cv, block, block_len, start_flag | end_flag);
        for i in 0..8 {
            cv[i] = root_state[i];
        }
    }

    let mut out = [0 as u8; 32];
    for i in 0..8 {
        let w = root_state[i];
        out[4 * i] = (w % 256) as u8;
        out[4 * i + 1] = ((w / 256) % 256) as u8;
        out[4 * i + 2] = ((w / 65536) % 256) as u8;
        out[4 * i + 3] = (w / 16777216) as u8;
    }
    out
}
