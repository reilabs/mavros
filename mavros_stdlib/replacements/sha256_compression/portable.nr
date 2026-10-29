use super::K;

fn rotr<let N: u32>(x: u32) -> u32 {
    // Mask before shifting: Mavros checks that the shifted value still fits u32.
    (x >> N) | ((x & ((1 << N) - 1)) << (32 - N))
}

fn add<let N: u32>(words: [u32; N]) -> u32 {
    // Five u32 words need 35 bits (five * (2^32 - 1) exceeds u34). Keep and truncate the carry,
    // avoiding the Field-to-integer conversion used by std's u32::wrapping_add.
    let mut sum: u35 = 0;
    for word in words {
        sum = sum + word as u35;
    }
    sum as u32
}

// SHA-256 works modulo 2^32. No spread word or sum is encoded as a Field,
// so this implementation does not rely on the proving field's modulus.
pub fn sha256_compression(msg_block: [u32; 16], state: [u32; 8]) -> [u32; 8] {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = msg_block[i];
    }
    for i in 16..64 {
        let x = w[i - 15];
        let y = w[i - 2];
        let s0 = rotr::<7>(x) ^ rotr::<18>(x) ^ (x >> 3);
        let s1 = rotr::<17>(y) ^ rotr::<19>(y) ^ (y >> 10);
        w[i] = add([w[i - 16], s0, w[i - 7], s1]);
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
        let s1 = rotr::<6>(e) ^ rotr::<11>(e) ^ rotr::<25>(e);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = add([h, s1, ch, K[i], w[i]]);
        let s0 = rotr::<2>(a) ^ rotr::<13>(a) ^ rotr::<22>(a);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = add([s0, maj]);
        h = g;
        g = f;
        f = e;
        e = add([d, t1]);
        d = c;
        c = b;
        b = a;
        a = add([t1, t2]);
    }

    [
        add([state[0], a]),
        add([state[1], b]),
        add([state[2], c]),
        add([state[3], d]),
        add([state[4], e]),
        add([state[5], f]),
        add([state[6], g]),
        add([state[7], h]),
    ]
}
