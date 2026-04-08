// SHA-256 Compression Function
// Translated from gnark-sha256 (github.com/reilabs/gnark-sha256)
// Uses the spread-table technique from Halo2

// ===== Round Constants =====

global K: [Field; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// ===== Spread Tables =====
// spread(x) places bit j of x at position 2*j, zeros at odd positions.

comptime fn make_spread_table<let N: u32>() -> [Field; N] {
    let mut table: [Field; N] = [0; N];
    let mut i: u32 = 0;
    for _ in 0..N {
        let mut spread: Field = 0;
        let mut x: u32 = i;
        let mut bit_val: Field = 1;
        for _ in 0..16 {
            if x % 2 == 1 {
                spread += bit_val;
            }
            x /= 2;
            bit_val *= 4;
        }
        table[i] = spread;
        i += 1;
    }
    table
}

global SPREAD_2: [Field; 4] = comptime { make_spread_table::<4>() };
global SPREAD_3: [Field; 8] = comptime { make_spread_table::<8>() };
global SPREAD_4: [Field; 16] = comptime { make_spread_table::<16>() };
global SPREAD_5: [Field; 32] = comptime { make_spread_table::<32>() };
global SPREAD_6: [Field; 64] = comptime { make_spread_table::<64>() };
global SPREAD_7: [Field; 128] = comptime { make_spread_table::<128>() };
global SPREAD_9: [Field; 512] = comptime { make_spread_table::<512>() };
global SPREAD_10: [Field; 1024] = comptime { make_spread_table::<1024>() };
global SPREAD_11: [Field; 2048] = comptime { make_spread_table::<2048>() };
global SPREAD_13: [Field; 8192] = comptime { make_spread_table::<8192>() };
global SPREAD_14: [Field; 16384] = comptime { make_spread_table::<16384>() };
global SPREAD_16: [Field; 65536] = comptime { make_spread_table::<65536>() };

fn spread_lookup(value: Field, size: u32) -> Field {
    // The array access implicitly range-checks value to [0, 2^size)
    if size == 2 { SPREAD_2[value as u32] }
    else if size == 3 { SPREAD_3[value as u32] }
    else if size == 4 { SPREAD_4[value as u32] }
    else if size == 5 { SPREAD_5[value as u32] }
    else if size == 6 { SPREAD_6[value as u32] }
    else if size == 7 { SPREAD_7[value as u32] }
    else if size == 9 { SPREAD_9[value as u32] }
    else if size == 10 { SPREAD_10[value as u32] }
    else if size == 11 { SPREAD_11[value as u32] }
    else if size == 13 { SPREAD_13[value as u32] }
    else if size == 14 { SPREAD_14[value as u32] }
    else { SPREAD_16[value as u32] }
}

// ===== Powers of 2 =====

global POW2: [Field; 65] = comptime {
    let mut table: [Field; 65] = [0; 65];
    table[0] = 1;
    for i in 1..65 {
        table[i] = table[i - 1] * 2;
    }
    table
};

// ===== Unconstrained Hint Functions =====

unconstrained fn hint_split4(x: u32, bcd: u32, cd: u32, d: u32) -> [u32; 4] {
    let seg_d = x & ((1 << d) - 1);
    let seg_c = (x >> d) & ((1 << (cd - d)) - 1);
    let seg_b = (x >> cd) & ((1 << (bcd - cd)) - 1);
    let seg_a = x >> bcd;
    [seg_a, seg_b, seg_c, seg_d]
}

unconstrained fn hint_split16(x: u32) -> [u32; 2] {
    [(x >> 16) & 0xFFFF, x & 0xFFFF]
}

unconstrained fn hint_unspread(x: u64) -> [u32; 4] {
    let mut e: u64 = 0;
    let mut o: u64 = 0;
    for i in 0..32 {
        let ii = i as u64;
        e |= ((x >> (2 * ii)) & 1) << ii;
        o |= ((x >> (2 * ii + 1)) & 1) << ii;
    }
    [((o >> 16) & 0xFFFF) as u32, (o & 0xFFFF) as u32, ((e >> 16) & 0xFFFF) as u32, (e & 0xFFFF) as u32]
}

unconstrained fn hint_add_decompose(sum: Field) -> [Field; 3] {
    let s = sum as u64;
    [(s & 0xFFFF) as Field, ((s >> 16) & 0xFFFF) as Field, (s >> 32) as Field]
}

unconstrained fn to_u64(x: Field) -> u64 {
    x as u64
}

// ===== Core Circuit Functions =====

fn split_and_spread4(x: Field, a: u32, b: u32, c: u32, d: u32) -> (Field, Field, Field, Field) {
    // Safety: constrained by reconstruction check and spread table lookups below
    let segs = unsafe { hint_split4(x as u32, b + c + d, c + d, d) };
    let sa = segs[0] as Field;
    let sb = segs[1] as Field;
    let sc = segs[2] as Field;
    let sd = segs[3] as Field;
    let rec = sa * POW2[b + c + d] + sb * POW2[c + d] + sc * POW2[d] + sd;
    assert(x == rec);
    (spread_lookup(sa, a), spread_lookup(sb, b), spread_lookup(sc, c), spread_lookup(sd, d))
}

fn spread(x: Field) -> Field {
    // Safety: constrained by reconstruction check and spread table lookups below
    let parts = unsafe { hint_split16(x as u32) };
    let hi = parts[0] as Field;
    let lo = parts[1] as Field;
    assert(x == hi * POW2[16] + lo);
    let hi_s = SPREAD_16[parts[0]];
    let lo_s = SPREAD_16[parts[1]];
    lo_s + hi_s * POW2[32]
}

fn unspread(x: Field) -> (Field, Field) {
    // Safety: constrained by reconstruction check and spread table lookups below
    let chunks = unsafe { hint_unspread(to_u64(x)) };
    let hi_odd = chunks[0] as Field;
    let lo_odd = chunks[1] as Field;
    let hi_even = chunks[2] as Field;
    let lo_even = chunks[3] as Field;
    let hi_odd_s = SPREAD_16[chunks[0]];
    let lo_odd_s = SPREAD_16[chunks[1]];
    let hi_even_s = SPREAD_16[chunks[2]];
    let lo_even_s = SPREAD_16[chunks[3]];
    let rec = lo_even_s + hi_even_s * POW2[32] + lo_odd_s * POW2[1] + hi_odd_s * POW2[33];
    assert(x == rec);
    let odd = lo_odd + hi_odd * POW2[16];
    let even = lo_even + hi_even * POW2[16];
    (odd, even)
}

fn unspread_into_spread(x: Field) -> (Field, Field) {
    // Safety: constrained by reconstruction check and spread table lookups below
    let chunks = unsafe { hint_unspread(to_u64(x)) };
    let hi_odd = chunks[0];
    let lo_odd = chunks[1];
    let hi_even = chunks[2];
    let lo_even = chunks[3];
    let hi_odd_s = SPREAD_16[hi_odd];
    let lo_odd_s = SPREAD_16[lo_odd];
    let hi_even_s = SPREAD_16[hi_even];
    let lo_even_s = SPREAD_16[lo_even];
    let rec = lo_even_s + hi_even_s * POW2[32] + lo_odd_s * POW2[1] + hi_odd_s * POW2[33];
    assert(x == rec);
    let odd = lo_odd_s + hi_odd_s * POW2[32];
    let even = lo_even_s + hi_even_s * POW2[32];
    (odd, even)
}

// ===== Modular Addition =====
// All SHA-256 additions are mod 2^32.

fn unsafe_add_n(values: [Field]) -> (Field, Field) {
    let mut sum: Field = 0;
    for i in 0..values.len() {
        sum += values[i];
    }
    // Safety: constrained by reconstruction check and spread table range checks below
    let parts = unsafe { hint_add_decompose(sum) };
    let lo = parts[0];
    let hi = parts[1];
    let carry = parts[2];
    // Range check lo and hi via 16-bit spread table lookup
    let _ = SPREAD_16[lo as u32];
    let _ = SPREAD_16[hi as u32];
    let res = lo + hi * POW2[16];
    assert(sum == res + carry * POW2[32]);
    (carry, res)
}

fn add2(a: Field, b: Field) -> Field {
    let (carry, res) = unsafe_add_n([a, b]);
    // carry in {0, 1}
    assert(carry * (carry - 1) == 0);
    res
}

fn add4(a: Field, b: Field, c: Field, d: Field) -> Field {
    let (carry, res) = unsafe_add_n([a, b, c, d]);
    // carry in {0, 1, 2, 3}: y = 2*carry - 3, y^2 in {1, 9}
    let y = 2 * carry - 3;
    let ysq = y * y;
    assert((ysq - 1) * (ysq - 9) == 0);
    res
}

fn add6(a: Field, b: Field, c: Field, d: Field, e: Field, f: Field) -> Field {
    let (carry, res) = unsafe_add_n([a, b, c, d, e, f]);
    // carry in {0..5}: y = 2*carry - 5, y^2 in {1, 9, 25}
    let y = 2 * carry - 5;
    let ysq = y * y;
    assert((ysq - 1) * (ysq - 9) * (ysq - 25) == 0);
    res
}

fn add7(a: Field, b: Field, c: Field, d: Field, e: Field, f: Field, g: Field) -> Field {
    let (carry, res) = unsafe_add_n([a, b, c, d, e, f, g]);
    // carry in {0..6}: y = carry - 3, y^2 in {0, 1, 4, 9}
    let y = carry - 3;
    let ysq = y * y;
    assert(ysq * (ysq - 1) * (ysq - 4) * (ysq - 9) == 0);
    res
}

// ===== Message Schedule =====

fn schedule(w_in: [Field; 16]) -> [Field; 64] {
    let mut w: [Field; 64] = [0; 64];
    for i in 0..16 {
        w[i] = w_in[i];
    }
    for i in 16..64 {
        // sigma0(w[i-15]): split into (14, 11, 4, 3) bits
        let (s14, s11, s4, s3) = split_and_spread4(w[i - 15], 14, 11, 4, 3);
        // ROTR7 in spread domain
        let rr7 = s11 + s14 * POW2[22] + s3 * POW2[50] + s4 * POW2[56];
        // ROTR18 in spread domain
        let rr18 = s14 + s3 * POW2[28] + s4 * POW2[34] + s11 * POW2[42];
        // SHR3 in spread domain
        let rs3 = s4 + s11 * POW2[8] + s14 * POW2[30];
        let s0_xor = rr7 + rr18 + rs3;
        let (_, s0) = unspread(s0_xor);

        // sigma1(w[i-2]): split into (13, 2, 7, 10) bits
        let (s13, s2, s7, s10) = split_and_spread4(w[i - 2], 13, 2, 7, 10);
        // ROTR17 in spread domain
        let rr17 = s2 + s13 * POW2[4] + s10 * POW2[30] + s7 * POW2[50];
        // ROTR19 in spread domain
        let rr19 = s13 + s10 * POW2[26] + s7 * POW2[46] + s2 * POW2[60];
        // SHR10 in spread domain
        let rs10 = s7 + s2 * POW2[14] + s13 * POW2[18];
        let s1_xor = rr17 + rr19 + rs10;
        let (_, s1) = unspread(s1_xor);

        w[i] = add4(w[i - 16], s0, w[i - 7], s1);
    }
    w
}

// ===== Compression (Permute) =====

// 0x5555555555555555 = sum of 2^(2i) for i in 0..31, used for NOT in spread domain
global ALL_EVENS: Field = 0x5555555555555555;

fn permute(h_in: [Field; 8], chunk: [Field; 16]) -> [Field; 8] {
    let w = schedule(chunk);
    let mut a = h_in[0];
    let mut b = h_in[1];
    let mut c = h_in[2];
    let mut d = h_in[3];
    let mut e = h_in[4];
    let mut f = h_in[5];
    let mut g = h_in[6];
    let mut h = h_in[7];

    for i in 0..64 {
        // Sigma1(e): split e into (7, 14, 5, 6) bits
        let (es7, es14, es5, es6) = split_and_spread4(e, 7, 14, 5, 6);
        // ROTR6
        let err6 = es5 + es14 * POW2[10] + es7 * POW2[38] + es6 * POW2[52];
        // ROTR11
        let err11 = es14 + es7 * POW2[28] + es6 * POW2[42] + es5 * POW2[54];
        // ROTR25
        let err25 = es7 + es6 * POW2[14] + es5 * POW2[26] + es14 * POW2[36];
        let big_s1_s = err6 + err11 + err25;
        let (_, big_s1) = unspread(big_s1_s);

        // Ch(e, f, g) = (e AND f) XOR (NOT e AND g)
        // Reconstruct spread(e) from segments
        let es = es6 + es5 * POW2[12] + es14 * POW2[22] + es7 * POW2[50];
        let fs = spread(f);
        // e AND f: odd bits of (spread(e) + spread(f))
        let (e_and_f, _) = unspread_into_spread(es + fs);
        let gs = spread(g);
        // NOT e AND g: odd bits of (NOT spread(e) + spread(g))
        let not_es = ALL_EVENS - es;
        let (not_e_and_g, _) = unspread_into_spread(not_es + gs);
        // ch = (e AND f) XOR (NOT e AND g): even bits of sum of the two spread ANDs
        let (_, ch) = unspread(e_and_f + not_e_and_g);

        // Sigma0(a): split a into (10, 9, 11, 2) bits
        let (as10, as9, as11, as2) = split_and_spread4(a, 10, 9, 11, 2);
        // ROTR2
        let arr2 = as11 + as9 * POW2[22] + as10 * POW2[40] + as2 * POW2[60];
        // ROTR13
        let arr13 = as9 + as10 * POW2[18] + as2 * POW2[38] + as11 * POW2[42];
        // ROTR22
        let arr22 = as10 + as2 * POW2[20] + as11 * POW2[24] + as9 * POW2[46];
        let big_s0_s = arr2 + arr13 + arr22;
        let (_, big_s0) = unspread(big_s0_s);

        // Maj(a, b, c) = majority function
        // spread(a) + spread(b) + spread(c): odd bits give majority
        let a_spread = as2 + as11 * POW2[4] + as9 * POW2[26] + as10 * POW2[44];
        let bs = spread(b);
        let cs = spread(c);
        let (maj, _) = unspread(a_spread + bs + cs);

        // State update
        let old_h = h;
        h = g;
        g = f;
        f = e;
        e = add6(d, old_h, big_s1, ch, K[i], w[i]);
        d = c;
        c = b;
        b = a;
        a = add7(old_h, big_s1, ch, K[i], w[i], big_s0, maj);
    }

    let mut result: [Field; 8] = [0; 8];
    result[0] = add2(h_in[0], a);
    result[1] = add2(h_in[1], b);
    result[2] = add2(h_in[2], c);
    result[3] = add2(h_in[3], d);
    result[4] = add2(h_in[4], e);
    result[5] = add2(h_in[5], f);
    result[6] = add2(h_in[6], g);
    result[7] = add2(h_in[7], h);
    result
}

// ===== Public Entry Point =====

pub fn sha256_compression(input: [u32; 16], state: [u32; 8]) -> [u32; 8] {
    // Convert u32 inputs to Field
    let mut block: [Field; 16] = [0; 16];
    for i in 0..16 {
        block[i] = input[i] as Field;
    }
    let mut h: [Field; 8] = [0; 8];
    for i in 0..8 {
        h[i] = state[i] as Field;
    }

    let result = permute(h, block);

    // Convert Field results back to u32
    let mut out: [u32; 8] = [0; 8];
    for i in 0..8 {
        out[i] = result[i] as u32;
    }
    out
}
