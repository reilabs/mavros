// Pure-Noir implementation of Noir's `derive_pedersen_generators` builtin.
// FIELD-ASSUMPTION: L5-embedded-curve
//
// This follows the backend's canonical generator derivation exactly: BLAKE3 hashes the domain,
// then a 64-byte preimage is formed from that 32-byte hash followed by 32 zero bytes. The
// big-endian generator index overwrites bytes 32-35; bytes 36-63 deliberately remain zero. A
// second pair of BLAKE3 hashes supplies an x-coordinate and sign bit for try-and-increment
// hash-to-curve on Grumpkin.

use crate::embedded_curve_ops::EmbeddedCurvePoint;
use super::blake3::blake3;

global TONELLI_SHANKS_S: u32 = 28;
global TONELLI_SHANKS_Q_LO: u128 = 0x181585d2833e84879b9709143e1f593f;
global TONELLI_SHANKS_Q_HI: u128 = 0x30644e72e131a029b85045b68;
global TONELLI_SHANKS_Q_PLUS_ONE_OVER_TWO_LO: u128 =
    0xc0ac2e9419f4243cdcb848a1f0faca0;
global TONELLI_SHANKS_Q_PLUS_ONE_OVER_TWO_HI: u128 = 0x183227397098d014dc2822db4;
global LEGENDRE_EXPONENT_LO: u128 = 0x9419f4243cdcb848a1f0fac9f8000000;
global LEGENDRE_EXPONENT_HI: u128 = 0x183227397098d014dc2822db40c0ac2e;

// These are the same fast paths as `bn254_blackbox_solver::derive_generators`: the first eight
// generators for `DEFAULT_DOMAIN_SEPARATOR`, plus the generator used by `pedersen_hash_length`.
// The coordinates come from that solver's generator test vectors; the domain hashes are BLAKE3 of
// the corresponding separator bytes. The default table is used only when the requested interval
// is wholly inside [0, 8); the length entry is used only for exactly one generator at index zero.
global DEFAULT_DOMAIN_HASH: [u8; 32] = [
    0xa3, 0x52, 0x4b, 0x24, 0x60, 0x99, 0xc5, 0x06, 0x3c, 0xdb, 0xe3, 0x60, 0x5b, 0x6d, 0xc9, 0xff,
    0x4b, 0x11, 0x65, 0x90, 0x09, 0xd3, 0xfb, 0x97, 0xef, 0x8f, 0x1c, 0x66, 0x74, 0xfb, 0x08, 0xeb,
];
global LENGTH_DOMAIN_HASH: [u8; 32] = [
    0xf7, 0x87, 0x94, 0x03, 0xcc, 0x4e, 0x0f, 0xa6, 0xac, 0x5c, 0x99, 0x79, 0xe8, 0x11, 0xd2, 0x62,
    0x46, 0x9e, 0xc9, 0x0d, 0x78, 0xc2, 0xb2, 0x85, 0x2f, 0x1c, 0x58, 0xf6, 0x23, 0x87, 0xd0, 0x9d,
];
global DEFAULT_GENERATORS_X: [Field; 8] = [
    0x083e7911d835097629f0067531fc15cafd79a89beecb39903f69572c636f4a5a,
    0x054aa86a73cb8a34525e5bbed6e43ba1198e860f5f3950268f71df4591bde402,
    0x1c44f2a5207c81c28a8321a5815ce8b1311024bbed131819bbdaf5a2ada84748,
    0x26d8b1160c6821a30c65f6cb47124afe01c29f4338f44d4a12c9fccf22fb6fb2,
    0x20ed9c6a1d27271c4498bfce0578d59db1adbeaa8734f7facc097b9b994fcf6e,
    0x0224a8abc6c8b8d50373d64cd2a1ab1567bf372b3b1f7b861d7f01257052d383,
    0x0f106f6d46bc904a5290542490b2f238775ff3c445b2f8f704c466655f460a2a,
    0x0298f2e42249f0519c8a8abd91567ebe016e480f219b8c19461d6a595cc33696,
];
global DEFAULT_GENERATORS_Y: [Field; 8] = [
    0x1a7f5efaad7f315c25a918f30cc8d7333fccab7ad7c90f14de81bcc528f9935d,
    0x209dcfbf2cfb57f9f6046f44d71ac6faf87254afc7407c04eb621a6287cac126,
    0x03aaee36e6422a1d0191632ac6599ae9eba5ac2c17a8c920aa3caf8b89c5f8a8,
    0x05c70c3b9c0d25a4c100e3a27bf3cc375f8af8cdd9498ec4089a823d7464caff,
    0x29cd7d370938b358c62c4a00f73a0d10aba7e5aaa04704a0713f891ebeb92371,
    0x2358629b90eafb299d6650a311e79914b0215eb0a790810b26da5a826726d711,
    0x29ab84d472f1d33f42fe09c47b8f7710f01920d6155250126731e486877bcf27,
    0x035bec4b8520a4ece27bd5aafabee3dfe1390d7439c419a8c55aceb207aac83b,
];

fn pow_256(base: Field, exponent_lo: u128, exponent_hi: u128) -> Field {
    let mut result = 1;
    let mut power = base;
    let mut lo = exponent_lo;
    let mut hi = exponent_hi;

    for _ in 0..128 {
        if lo % 2 == 1 {
            result = result * power;
        }
        power = power * power;
        lo = lo / 2;
    }
    for _ in 0..128 {
        if hi % 2 == 1 {
            result = result * power;
        }
        power = power * power;
        hi = hi / 2;
    }
    result
}

// The BN254 scalar field has p - 1 = q * 2^28. Five is a quadratic non-residue.
fn sqrt(value: Field) -> (bool, Field) {
    if value == 0 {
        (true, 0)
    } else if pow_256(value, LEGENDRE_EXPONENT_LO, LEGENDRE_EXPONENT_HI) != 1 {
        (false, 0)
    } else {
        let mut c = pow_256(5, TONELLI_SHANKS_Q_LO, TONELLI_SHANKS_Q_HI);
        let mut x = pow_256(
            value,
            TONELLI_SHANKS_Q_PLUS_ONE_OVER_TWO_LO,
            TONELLI_SHANKS_Q_PLUS_ONE_OVER_TWO_HI,
        );
        let mut t = pow_256(value, TONELLI_SHANKS_Q_LO, TONELLI_SHANKS_Q_HI);
        let mut m = TONELLI_SHANKS_S;

        for _ in 0..TONELLI_SHANKS_S {
            if t != 1 {
                let mut t_squared = t;
                let mut i = 0;
                let mut found = false;
                for j in 1..TONELLI_SHANKS_S {
                    t_squared = t_squared * t_squared;
                    if !found & (j < m) & (t_squared == 1) {
                        i = j;
                        found = true;
                    }
                }
                assert(found);

                let squarings = m - i - 1;
                let mut b = c;
                for j in 0..TONELLI_SHANKS_S {
                    if j < squarings {
                        b = b * b;
                    }
                }

                x = x * b;
                let b_squared = b * b;
                t = t * b_squared;
                c = b_squared;
                m = i;
            }
        }
        assert_eq(t, 1);
        (true, x)
    }
}

fn digest_pair_to_field(high: [u8; 32], low: [u8; 32]) -> Field {
    // Interpreting the 64 digest bytes as one big-endian integer and accumulating in Field
    // performs the same reduction modulo p as the backend's 512-bit conversion.
    let mut value = 0;
    for i in 0..32 {
        value = value * 256 + high[i] as Field;
    }
    for i in 0..32 {
        value = value * 256 + low[i] as Field;
    }
    value
}

fn hash_to_curve(seed: [u8; 64], starting_attempt: u8) -> EmbeddedCurvePoint {
    let mut input = [0 as u8; 66];
    for i in 0..64 {
        input[i] = seed[i];
    }

    let mut point = EmbeddedCurvePoint::point_at_infinity();
    let mut found = false;
    let mut attempt = starting_attempt;
    for _ in 0..256 {
        if !found {
            input[64] = attempt;
            input[65] = 0;
            let high = blake3(input);
            input[65] = 1;
            let low = blake3(input);

            let x = digest_pair_to_field(high, low);
            let (is_square, root) = sqrt(x * x * x - 17);
            if is_square {
                let sign = high[0] > 127;
                let y = if root.sgn0() != sign { -root } else { root };
                point = EmbeddedCurvePoint { x, y };
                found = true;
            } else if attempt < 255 {
                attempt = attempt + 1;
            }
        }
    }
    assert(found);
    point
}

pub fn derive_pedersen_generators<let N: u32, let M: u32>(
    domain_separator_bytes: [u8; M],
    starting_index: u32,
) -> [EmbeddedCurvePoint; N] {
    let domain_hash = blake3(domain_separator_bytes);
    let mut generators = [EmbeddedCurvePoint::point_at_infinity(); N];

    // These additions intentionally retain Noir's checked-u32 trapping behaviour: an overflowing
    // interval is rejected instead of wrapping around to a different generator sequence.
    if (domain_hash == DEFAULT_DOMAIN_HASH) & (starting_index + N <= 8) {
        for i in 0..N {
            let index = starting_index + i;
            generators[i] = EmbeddedCurvePoint {
                x: DEFAULT_GENERATORS_X[index],
                y: DEFAULT_GENERATORS_Y[index],
            };
        }
    } else if (domain_hash == LENGTH_DOMAIN_HASH) & (starting_index == 0) & (N == 1) {
        generators[0] = EmbeddedCurvePoint {
            x: 0x2df8b940e5890e4e1377e05373fae69a1d754f6935e6a780b666947431f2cdcd,
            y: 0x2ecd88d15967bc53b885912e0d16866154acb6aac2d3f85e27ca7eefb2c19083,
        };
    } else {
        for i in 0..N {
            let index = starting_index + i;
            let mut seed = [0 as u8; 64];
            for j in 0..32 {
                seed[j] = domain_hash[j];
            }
            seed[32] = (index / 16777216) as u8;
            seed[33] = ((index / 65536) % 256) as u8;
            seed[34] = ((index / 256) % 256) as u8;
            seed[35] = (index % 256) as u8;
            generators[i] = hash_to_curve(seed, 0);
        }
    }
    generators
}
