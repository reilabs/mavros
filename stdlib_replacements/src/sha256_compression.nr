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

// Pre-decomposed word with sigma0 chunk layout [10, 9, 11, 2]
struct Sigma0Word {
    packed: u32,
    s0: Field, // spread of top 10 bits
    s1: Field, // spread of next 9 bits
    s2: Field, // spread of next 11 bits
    s3: Field, // spread of bottom 2 bits
}

impl Sigma0Word {
    // Implicitly rangechecks x because it uses split_and_spread_4
    fn new(x: u32) -> Self {
        let (s0, s1, s2, s3) = split_and_spread_4::<10,9,11,2>(x);
        Sigma0Word { packed: x, s0, s1, s2, s3 }
    }

    // Reconstruct full 64-bit spread from sigma0 chunks [10|9|11|2]
    // Bit offsets: s3 at 0, s2 at 2*2=4, s1 at 2*(2+11)=26, s0 at 2*(2+11+9)=44
    fn spread(self) -> Field {
        self.s3 + self.s2 * pow2(4) + self.s1 * pow2(26) + self.s0 * pow2(44)
    }

    // Sigma0(a) = ROTR2(a) XOR ROTR13(a) XOR ROTR22(a)
    fn big_sigma(self) -> u32 {
        // s0=as10, s1=as9, s2=as11, s3=as2
        let arr2 = self.s2 + self.s1 * pow2(22) + self.s0 * pow2(40) + self.s3 * pow2(60);
        let arr13 = self.s1 + self.s0 * pow2(18) + self.s3 * pow2(38) + self.s2 * pow2(42);
        let arr22 = self.s0 + self.s3 * pow2(20) + self.s2 * pow2(24) + self.s1 * pow2(46);
        // Safety: sum of 3 rotated spread values fits in 66 bits; unspread_64 rangechecks via lookups.
        let (_, S0) = unspread_64(unsafe_cast(arr2 + arr13 + arr22));
        S0
    }

    // Maj(a,b,c) = majority of each bit, from pre-decomposed chunks
    fn maj(a: Self, b: Self, c: Self) -> u32 {
        // Safety: this will not overflow by construction - for each input, the biggest 2-bit-aligned chunk is 0b01,
        // so the max sum is 0b11, so each 2-bit chunk is guaranteed to not have a carry.
        let (maj, _) = unspread_64(unsafe_cast(
            a.spread() + b.spread() + c.spread(),
        ));
        maj
    }

    fn wrapping_add<let N: u32>(a: [u32; N]) -> Self {
        // Safety: calls `new` which does a rangecheck.
        Sigma0Word::new(unsafe_cast(unsafe_wrapping_add_N(a)))
    }
}

// Pre-decomposed word with sigma1 chunk layout [7, 14, 5, 6]
struct Sigma1Word {
    packed: u32,
    s0: Field, // spread of top 7 bits
    s1: Field, // spread of next 14 bits
    s2: Field, // spread of next 5 bits
    s3: Field, // spread of bottom 6 bits
}

impl Sigma1Word {
    // Implicitly rangechecks x, because it uses split_and_spread_4.
    fn new(x: u32) -> Self {
        let (s0, s1, s2, s3) = split_and_spread_4::<7,14,5,6>(x);
        Sigma1Word { packed: x, s0, s1, s2, s3 }
    }

    // Reconstruct full 64-bit spread from sigma1 chunks [7|14|5|6]
    // Bit offsets: s3 at 0, s2 at 2*6=12, s1 at 2*(6+5)=22, s0 at 2*(6+5+14)=50
    fn spread(self) -> Field {
        self.s3 + self.s2 * pow2(12) + self.s1 * pow2(22) + self.s0 * pow2(50)
    }

    // Sigma1(e) = ROTR6(e) XOR ROTR11(e) XOR ROTR25(e)
    fn big_sigma(self) -> u32 {
        let err6 = self.s2 + self.s1 * pow2(10) + self.s0 * pow2(38) + self.s3 * pow2(52);
        let err11 = self.s1 + self.s0 * pow2(28) + self.s3 * pow2(42) + self.s2 * pow2(54);
        let err25 = self.s0 + self.s3 * pow2(14) + self.s2 * pow2(26) + self.s1 * pow2(36);
        // Safety: this will not overflow by construction - for each input, the biggest 2-bit-aligned chunk is 0b01,
        // so the max sum is 0b11, so each 2-bit chunk is guaranteed to not have a carry.
        let (_, S1) = unspread_64(unsafe_cast(err6 + err11 + err25));
        S1
    }

    // Ch(e,f,g) = (e & f) XOR (~e & g) = (e&f) + g - (e&g)
    // In we have for each bit i: `g_i <= (e&g)_i`. Moreover, we have g_i - (e&g)_i = 1 iff `g_i = 1` and `(e&g)_i = 0`, so `e_i = 0`.
    // In other words, g_i - (e&g)_i = g_i & (~e_i).
    // Moreover, e&f and (~e&g) are disjoint, so XOR can be replaced by addition.
    fn ch(e: Self, f: Self, g: Self) -> u32 {
        let e_spread = e.spread();
        let f_spread = f.spread();
        let g_spread = g.spread();
        // Safety: this will not overflow by construction - for each input, the biggest 2-bit-aligned chunk is 0b01,
        // so the max sum is 0b10, so each 2-bit chunk is guaranteed to not have a carry.
        let (e_and_f, _) = unspread_64(unsafe_cast(e_spread + f_spread));
        let (e_and_g, _) = unspread_64(unsafe_cast(e_spread + g_spread));
        // Safety: this is just an algebraic encoding of the ch function, so it will not overflow by construction.
        unsafe_cast(e_and_f as Field + g.packed as Field - e_and_g as Field)
    }

    fn wrapping_add<let N: u32>(a: [u32; N]) -> Self {
        // Safety: calls `new` which does a rangecheck.
        Sigma1Word::new(unsafe_cast(unsafe_wrapping_add_N(a)))
    }
}

unconstrained fn split4_hint(f: u32, a: u32, b: u32, c: u32, d: u32) -> (Field, Field, Field) {
    let a0 = (f & (((1 << a) - 1) << (b + c + d))) >> (b + c + d);
    let a1 = (f & (((1 << b) - 1) << (c + d))) >> (c + d);
    let a2 = (f & (((1 << c) - 1) << d)) >> d;
    (a0 as Field, a1 as Field, a2 as Field)
}

// Implicitly rangechecks f: the spread lookups on each chunk fail if any chunk is out of range,
// which constrains f to be a valid u32 decomposable into chunks of widths [a, b, c, d].
fn split_and_spread_4<let a: u32, let b: u32, let c: u32, let d: u32>(f: u32) -> (Field, Field, Field, Field) {
    assert(a + b + c + d == 32);
    // Safety: the definition of `a3` ensures that `a0`, `a1`, `a2`, and `a3` form the proper decomposition of `f`. The calls to `spread`
    // ensure that each chunk is exactly in the defined range.
    let (a0, a1, a2) = unsafe { split4_hint(f, a, b, c, d) };
    let a3 = (f as Field) - (a0 * ((1 << (b + c + d)) as Field)) - (a1 * ((1 << (c + d)) as Field)) - (a2 * ((1 << d) as Field));
    // Safety: each chunk is constrained to its bit-width by the spread lookup (spread<N> fails if input >= 2^N).
    (spread::<a>(unsafe_cast(a0)) as Field, spread::<b>(unsafe_cast(a1)) as Field, spread::<c>(unsafe_cast(a2)) as Field, spread::<d>(unsafe_cast(a3)) as Field)
}

unconstrained fn unspread_64_hint(f: u64) -> (Field, Field) {
    let mask22: u64 = (1 << 22) - 1;
    let a0 = (f >> 42) & mask22;
    let a1 = (f >> 20) & mask22;
    (a0 as Field, a1 as Field)
}

// Implicitly rangechecks x: the unspread lookups on each limb fail if any limb is out of range,
// which constrains x to be a valid 64-bit spread value decomposable into 3 limbs of [22, 22, 20] spread bits.
fn unspread_64(x: u64) -> (u32, u32) {
    // Safety: the definition of `a2` ensures that `a0`, `a1`, and `a2` form the proper decomposition of `x`. The calls to `unspread`
    // ensure that each chunk is exactly in the defined range.
    let (a0, a1) = unsafe { unspread_64_hint(x) };
    let a2 = (x as Field) - (a0 * pow2(42)) - (a1 * pow2(20));
    // Safety: each limb is constrained to its bit-width by the unspread lookup (unspread<N> fails if input is not a valid 2N-bit spread).
    let (a0_uns_0, a0_uns_1) = unspread::<11>(unsafe_cast(a0));
    let (a1_uns_0, a1_uns_1) = unspread::<11>(unsafe_cast(a1));
    let (a2_uns_0, a2_uns_1) = unspread::<10>(unsafe_cast(a2));
    let a_0 = a2_uns_0 as Field + a1_uns_0 as Field * pow2(10) + a0_uns_0 as Field * pow2(21);
    let a_1 = a2_uns_1 as Field + a1_uns_1 as Field * pow2(10) + a0_uns_1 as Field * pow2(21);
    // Safety: each component is rangechecked by unspread, and the weighted sum fits in 32 bits
    // (10 + 11 + 11 = 32). Consumer (wrapping_add -> split_and_spread_4) also rangechecks.
    (unsafe_cast(a_0), unsafe_cast(a_1))
}

fn small_sigma0(x: u32) -> u32 {
    let (wi15s14, wi15s11, wi15s4, wi15s3) = split_and_spread_4::<14,11,4,3>(x);
    let wi15rr7 = wi15s11 + wi15s14 * pow2(22) + wi15s3 * pow2(50) + wi15s4 * pow2(56);
    let wi15rr18 = wi15s14 + wi15s3 * pow2(28) + wi15s4 * pow2(34) + wi15s11 * pow2(42);
    let wi15rs3 = wi15s4 + wi15s11 * pow2(8) + wi15s14 * pow2(30);
    let s0XorPlus = wi15rr7 + wi15rr18 + wi15rs3;
    // Safety: this will not overflow by construction - for each input, the biggest 2-bit-aligned chunk is 0b01,
    // so the max sum is 0b11, so each 2-bit chunk is guaranteed to not have a carry.
    let (_, s0) = unspread_64(unsafe_cast(s0XorPlus));
    s0
}

fn small_sigma1(x: u32) -> u32 {
    let (wi2s13, wi2s2, wi2s7, wi2s10) = split_and_spread_4::<13,2,7,10>(x);
    let wi2rr17 = wi2s2 + wi2s13 * pow2(4) + wi2s10 * pow2(30) + wi2s7 * pow2(50);
    let wi2rr19 = wi2s13 + wi2s10 * pow2(26) + wi2s7 * pow2(46) + wi2s2 * pow2(60);
    let wi2rs10 = wi2s7 + wi2s2 * pow2(14) + wi2s13 * pow2(18);
    let s1XorPlus = wi2rr17 + wi2rr19 + wi2rs10;
    // Safety: this will not overflow by construction - for each input, the biggest 2-bit-aligned chunk is 0b01,
    // so the max sum is 0b11, so each 2-bit chunk is guaranteed to not have a carry.
    let (_, s1) = unspread_64(unsafe_cast(s1XorPlus));
    s1
}

unconstrained fn wrapping_add_N_hint(a: Field) -> Field {
    (a as u32) as Field
}

// Result must be rangechecked later!
fn unsafe_wrapping_add_N<let N: u32>(a: [u32; N]) -> Field {
    assert(N < 20);
    assert(N > 0);
    let mut sum: Field = 0;
    for i in a {
        sum = sum + i as Field;
    }
    // Safety: this is unsafe - we know res and carry form a proper decomposition and carry is small,
    // but res needs rangechecking.
    let res = unsafe { wrapping_add_N_hint(sum) };
    let carry = (sum - res as Field) / pow2(32);
    carry.assert_max_bit_size::<8>();
    res as Field
}

fn wrapping_add_N<let N: u32>(a: [u32; N]) -> u32 {
    let res = unsafe_wrapping_add_N(a);
    // Explicit rangecheck since this is the terminal consumer (no spread/unspread downstream).
    res.assert_max_bit_size::<32>();
    // Safety: checked above by the rangecheck.
    unsafe_cast(res)
}

fn pow2(n: u64) -> Field {
    (1u64 << n) as Field
}

pub fn sha256_compression(msg_block: [u32; 16], state: [u32; 8]) -> [u32; 8] {
    // Message schedule
    let mut w: [u32; 64] = [0; 64];
    for i in 0..16 {
        w[i] = msg_block[i];
    }
    for i in 16..64 {
        w[i] = wrapping_add_N([small_sigma1(w[i - 2]), w[i - 7], small_sigma0(w[i - 15]), w[i - 16]]);
    }

    // Pre-decompose initial state into spread chunks.
    // a,b,c use sigma0 layout (for big_sigma + maj reuse).
    // e,f,g use sigma1 layout (for big_sigma + ch reuse).
    // d,h are only used in additions so stay as plain u32.
    let mut a_w = Sigma0Word::new(state[0]);
    let mut b_w = Sigma0Word::new(state[1]);
    let mut c_w = Sigma0Word::new(state[2]);
    let mut d: u32 = state[3];
    let mut e_w = Sigma1Word::new(state[4]);
    let mut f_w = Sigma1Word::new(state[5]);
    let mut g_w = Sigma1Word::new(state[6]);
    let mut h: u32 = state[7];

    for i in 0..64 {
        // Reuse cached spread chunks - no re-decomposition needed
        let bs1 = e_w.big_sigma();
        let ch = Sigma1Word::ch(e_w, f_w, g_w);
        let bs0 = a_w.big_sigma();
        let maj = Sigma0Word::maj(a_w, b_w, c_w);

        // Fused add + decompose: result is pre-decomposed for next round
        let new_e_w = Sigma1Word::wrapping_add([d, h, bs1, ch, K[i], w[i]]);
        let new_a_w = Sigma0Word::wrapping_add([h, bs1, ch, K[i], w[i], bs0, maj]);

        // Rotate working variables, carrying decompositions forward
        h = g_w.packed;
        g_w = f_w;
        f_w = e_w;
        e_w = new_e_w;
        d = c_w.packed;
        c_w = b_w;
        b_w = a_w;
        a_w = new_a_w;
    }

    [
        wrapping_add_N([state[0], a_w.packed]),
        wrapping_add_N([state[1], b_w.packed]),
        wrapping_add_N([state[2], c_w.packed]),
        wrapping_add_N([state[3], d]),
        wrapping_add_N([state[4], e_w.packed]),
        wrapping_add_N([state[5], f_w.packed]),
        wrapping_add_N([state[6], g_w.packed]),
        wrapping_add_N([state[7], h]),
    ]
}
