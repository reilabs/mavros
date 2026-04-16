use dep::bignum::RuntimeBigNum;

fn compare_signature_sha256<let N: u32>(padded_sha256_hash: [u8; N], msg_hash: [u8; 32]) -> bool {
    for i in 0..32 {
        assert(padded_sha256_hash[31 - i] == msg_hash[i]);
    }

    let hash_prefix: [u8; 19] =
        [32, 4, 0, 5, 1, 2, 4, 3, 101, 1, 72, 134, 96, 9, 6, 13, 48, 49, 48];

    for i in 32..51 {
        assert(hash_prefix[i - 32] == padded_sha256_hash[i]);
    }

    assert(padded_sha256_hash[51] == 0);

    let ps_len = N - 54;
    for i in 52..N {
        if i < 52 + ps_len {
            assert(padded_sha256_hash[i] == 255);
        } else if i == 52 + ps_len {
            assert(padded_sha256_hash[i] == 1);
        } else if i == 53 + ps_len {
            assert(padded_sha256_hash[i] == 0);
        } else {
            assert(padded_sha256_hash[i] == 0);
        }
    }

    true
}

fn exponentiate_signature<let NumLimbs: u32, let ModBits: u32>(
    sig: RuntimeBigNum<NumLimbs, ModBits>,
    exponent: u32,
) -> RuntimeBigNum<NumLimbs, ModBits> {
    assert((exponent > 0) & (exponent < 131072), "Exponent must be positive and less than 2^17");

    let mut result = RuntimeBigNum::one(sig.params);
    let mut base = sig;
    let mut exp = exponent;

    for _ in 0..17 {
        if exp > 0 {
            if exp % 2 == 1 {
                result = result * base;
            }
            base = base * base;
            exp = exp / 2;
        }
    }

    result
}

pub fn verify_sha256_pkcs1v15<let NumLimbs: u32, let ModBits: u32>(
    msg_hash: [u8; 32],
    sig: RuntimeBigNum<NumLimbs, ModBits>,
    exponent: u32,
) -> bool {
    let exponentiated = exponentiate_signature(sig, exponent);
    let padded_sha256_hash_bytes: [u8; (ModBits + 7) / 8] = exponentiated.to_le_bytes();
    compare_signature_sha256(padded_sha256_hash_bytes, msg_hash)
}
