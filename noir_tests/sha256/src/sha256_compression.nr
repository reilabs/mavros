// SHA-256 compression function replacement
// Replaces the stdlib sha256_compression lowlevel intrinsic.

use std::mavros::{spread, unspread, unsafe_cast};

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

unconstrained fn split4_hint(f: u32, a: u32, b: u32, c: u32, d: u32) -> (Field, Field, Field) {
    let a0 = (f & (((1 << a) - 1) << (b + c + d))) >> (b + c + d);
    let a1 = (f & (((1 << b) - 1) << (c + d))) >> (c + d);
    let a2 = (f & (((1 << c) - 1) << d)) >> d;
    (a0 as Field, a1 as Field, a2 as Field)
}

fn split_and_spread_4<let a: u32, let b: u32, let c: u32, let d: u32>(f: u32) -> (Field, Field, Field, Field) {
    assert(a + b + c + d == 32);
    // Safety: is for the weak
    let (a0, a1, a2) = unsafe { split4_hint(f, a, b, c, d) };
    let a3 = (f as Field) - (a0 * ((1 << (b + c + d)) as Field)) - (a1 * ((1 << (c + d)) as Field)) - (a2 * ((1 << d) as Field));
    (spread::<a>(unsafe_cast(a0)) as Field, spread::<b>(unsafe_cast(a1)) as Field, spread::<c>(unsafe_cast(a2)) as Field, spread::<d>(unsafe_cast(a3)) as Field)
}

unconstrained fn unspread_64_hint(f: u64) -> (Field, Field) {
    let a = 22;
    let b = 22;
    let c = 20;
    let a0 = (f & (((1 << a) - 1) << (b + c))) >> (b + c);
    let a1 = (f & (((1 << b) - 1) << (c))) >> (c);
    (a0 as Field, a1 as Field)
}

fn unspread_64(x: u64) -> (u32, u32) {
    // Safety: is for the weak
    let (a0, a1) = unsafe { unspread_64_hint(x) };
    let b = 22;
    let c = 20;
    let a2 = (x as Field) - (a0 * ((1 << (b + c)) as Field)) - (a1 * ((1 << c) as Field));
    let (a0_uns_0, a0_uns_1) = unspread::<11>(unsafe_cast(a0));
    let (a1_uns_0, a1_uns_1) = unspread::<11>(unsafe_cast(a1));
    let (a2_uns_0, a2_uns_1) = unspread::<10>(unsafe_cast(a2));
    let a_0 = a0_uns_0 as Field + a1_uns_0 as Field * POW_2_11 + a2_uns_0 as Field * POW_2_21;
    let a_1 = a0_uns_1 as Field + a1_uns_1 as Field * POW_2_11 + a2_uns_1 as Field * POW_2_21;
    (unsafe_cast(a_0), unsafe_cast(a_1))
}

fn small_sigma0(x: u32) -> u32 {
    let (wi15s14, wi15s11, wi15s4, wi15s3) = split_and_spread_4::<14,11,4,3>(x);
    let wi15rr7 = wi15s11 + wi15s14 * POW_2_22 + wi15s3 * POW_2_50 + wi15s4 * POW_2_56;
    let wi15rr18 = wi15s14 + wi15s3 * POW_2_28 + wi15s4 * POW_2_34 + wi15s11 * POW_2_42;
    let wi15rs3 = wi15s4 + wi15s11 * POW_2_8 + wi15s14 * POW_2_30;
    let s0XorPlus = wi15rr7 + wi15rr18 + wi15rs3;
    let (_, s0) = unspread_64(unsafe_cast(s0XorPlus));
    s0
}

fn small_sigma1(x: u32) -> u32 {
    // rotr(17) ^ rotr(19) ^ shr(10)
    rotr(x, 131072, 32768) ^ rotr(x, 524288, 8192) ^ (x / 1024)
}

fn wrapping_add(a: u32, b: u32) -> u32 {
    let sum = a as u64 + b as u64;
    (sum & 0xffffffff) as u32
}

global POW_2_22: Field = (1u64 << 22) as Field;
global POW_2_11: Field = (1u64 << 11) as Field;
global POW_2_28: Field = (1u64 << 28) as Field;
global POW_2_21: Field = (1u64 << 21) as Field;
global POW_2_50: Field = (1u64 << 50) as Field;
global POW_2_56: Field = (1u64 << 56) as Field;
global POW_2_34: Field = (1u64 << 34) as Field;
global POW_2_42: Field = (1u64 << 42) as Field;
global POW_2_30: Field = (1u64 << 30) as Field;
global POW_2_8: Field = (1u64 << 8) as Field;



pub fn sha256_compression(msg_block: [u32; 16], state: [u32; 8]) -> [u32; 8] {
    // Message schedule
    let mut w: [u32; 64] = [0; 64];
    for i in 0..16 {
        w[i] = msg_block[i];
    }
    for i in 16..64 {
		// wi15rr7 := hasher.api.Add(
		// 	wi15s11,
		// 	hasher.api.Mul(wi15s14, 1<<22),
		// 	hasher.api.Mul(wi15s3, 1<<50),
		// 	hasher.api.Mul(wi15s4, 1<<56),
		// )
		// wi15rr18 := hasher.api.Add(
		// 	wi15s14,
		// 	hasher.api.Mul(wi15s3, 1<<28),
		// 	hasher.api.Mul(wi15s4, 1<<34),
		// 	hasher.api.Mul(wi15s11, 1<<42),
		// )
		// wi15rs3 := hasher.api.Add(
		// 	wi15s4,
		// 	hasher.api.Mul(wi15s11, 1<<8),
		// 	hasher.api.Mul(wi15s14, 1<<30),
		// )
		// s0XorPlus := hasher.api.Add(wi15rr7, wi15rr18, wi15rs3)
		// _, s0 := hasher.unspread(s0XorPlus)

        
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
