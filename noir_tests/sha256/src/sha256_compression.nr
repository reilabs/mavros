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

fn ch(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ (!e & g)
}

fn maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}

fn big_sigma0(a: u32) -> u32 {
    let (as10, as9, as11, as2) = split_and_spread_4::<10,9,11,2>(a);
    let arr2 = as11 + as9 * POW_2_22 + as10 * POW_2_40 + as2 * POW_2_60;
    let arr13 = as9 + as10 * POW_2_18 + as2 * POW_2_38 + as11 * POW_2_42;
    let arr22 = as10 + as2 * POW_2_20 + as11 * POW_2_24 + as9 * POW_2_46;
    let S0S = arr2 + arr13 + arr22;
    let (_, S0) = unspread_64(unsafe_cast(S0S));
    S0
}

fn big_sigma1(e: u32) -> u32 {
    let (es7, es14, es5, es6) = split_and_spread_4::<7,14,5,6>(e);
    let err6 = es5 + es14 * POW_2_10 + es7 * POW_2_38 + es6 * POW_2_52;
    let err11 = es14 + es7 * POW_2_28 + es6 * POW_2_42 + es5 * POW_2_54;
    let err25 = es7 + es6 * POW_2_14 + es5 * POW_2_26 + es14 * POW_2_36;
    let S1S = err6 + err11 + err25;
    let (_, S1) = unspread_64(unsafe_cast(S1S));
    S1
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
    let mask22: u64 = (1 << 22) - 1;
    let a0 = (f >> 42) & mask22;
    let a1 = (f >> 20) & mask22;
    (a0 as Field, a1 as Field)
}

fn unspread_64(x: u64) -> (u32, u32) {
    // Safety: is for the weak
    let (a0, a1) = unsafe { unspread_64_hint(x) };
    let a2 = (x as Field) - (a0 * POW_2_42) - (a1 * POW_2_20);
    let (a0_uns_0, a0_uns_1) = unspread::<11>(unsafe_cast(a0));
    let (a1_uns_0, a1_uns_1) = unspread::<11>(unsafe_cast(a1));
    let (a2_uns_0, a2_uns_1) = unspread::<10>(unsafe_cast(a2));
    let a_0 = a2_uns_0 as Field + a1_uns_0 as Field * POW_2_10 + a0_uns_0 as Field * POW_2_21;
    let a_1 = a2_uns_1 as Field + a1_uns_1 as Field * POW_2_10 + a0_uns_1 as Field * POW_2_21;
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
    let (wi2s13, wi2s2, wi2s7, wi2s10) = split_and_spread_4::<13,2,7,10>(x);
    let wi2rr17 = wi2s2 + wi2s13 * POW_2_4 + wi2s10 * POW_2_30 + wi2s7 * POW_2_50;
    let wi2rr19 = wi2s13 + wi2s10 * POW_2_26 + wi2s7 * POW_2_46 + wi2s2 * POW_2_60;
    let wi2rs10 = wi2s7 + wi2s2 * POW_2_14 + wi2s13 * POW_2_18;
    let s1XorPlus = wi2rr17 + wi2rr19 + wi2rs10;
    let (_, s1) = unspread_64(unsafe_cast(s1XorPlus));
    s1
}

unconstrained fn wrapping_add_N_hint(a: Field) -> u32 {
    a as u32
}

fn wrapping_add_N<let N: u32>(a: [u32; N]) -> u32 {
    assert(N < 20);
    assert(N > 0);
    let mut sum: Field = 0;
    for i in a {
        sum = sum + i as Field;
    }

    // Safety: is for the weak
    let res = unsafe { wrapping_add_N_hint(sum) };
    let carry = (sum - res as Field) / POW_2_32;
    carry.assert_max_bit_size::<8>();
    (N as Field - 1 - carry).assert_max_bit_size::<8>();
    res
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
global POW_2_20: Field = (1u64 << 20) as Field;
global POW_2_8: Field = (1u64 << 8) as Field;
global POW_2_10: Field = (1u64 << 10) as Field;
global POW_2_4: Field = (1u64 << 4) as Field;
global POW_2_26: Field = (1u64 << 26) as Field;
global POW_2_46: Field = (1u64 << 46) as Field;
global POW_2_60: Field = (1u64 << 60) as Field;
global POW_2_14: Field = (1u64 << 14) as Field;
global POW_2_18: Field = (1u64 << 18) as Field;
global POW_2_32: Field = (1u64 << 32) as Field;
global POW_2_38: Field = (1u64 << 38) as Field;
global POW_2_52: Field = (1u64 << 52) as Field;
global POW_2_54: Field = (1u64 << 54) as Field;
global POW_2_36: Field = (1u64 << 36) as Field;
global POW_2_40: Field = (1u64 << 40) as Field;
global POW_2_24: Field = (1u64 << 24) as Field;

pub fn sha256_compression(msg_block: [u32; 16], state: [u32; 8]) -> [u32; 8] {
    // Message schedule
    let mut w: [u32; 64] = [0; 64];
    for i in 0..16 {
        w[i] = msg_block[i];
    }
    for i in 16..64 {        
        w[i] = wrapping_add_N([small_sigma1(w[i - 2]), w[i - 7], small_sigma0(w[i - 15]), w[i - 16]]);
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

        let bs1 = big_sigma1(e);
        let bs0 = big_sigma0(a);
        let ch = ch(e, f, g);
        let k = K[i];
        let w = w[i];
        let maj = maj(a, b, c);

        let t1 = wrapping_add_N([
            h,
            bs1,
            ch,
            k,
            w,
        ]);

        h = g;
        g = f;
        f = e;
        e = wrapping_add_N([d, t1]);
        d = c;
        c = b;
        b = a;
        a = wrapping_add_N([t1, bs0, maj]);
    }

    [
        wrapping_add_N([state[0], a]),
        wrapping_add_N([state[1], b]),
        wrapping_add_N([state[2], c]),
        wrapping_add_N([state[3], d]),
        wrapping_add_N([state[4], e]),
        wrapping_add_N([state[5], f]),
        wrapping_add_N([state[6], g]),
        wrapping_add_N([state[7], h]),
    ]
}
