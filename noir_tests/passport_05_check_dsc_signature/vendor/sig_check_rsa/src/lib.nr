use bignum::{params::BigNumParams, RuntimeBigNum};
use common::sha256_and_check_data_to_sign;
use rsa::rsa::verify_sha256_pkcs1v15;

pub fn verify_signature<let SIG_BYTES: u32, let IS_PSS: u32, let DATA_TO_SIGN_MAX_LEN: u32, let HASH_BYTE_SIZE: u32>(
    pubkey_bytes: [u8; SIG_BYTES],
    // This is equivalent to sig_bytes: [u8; SIG_BYTES] but because of
    // an issue with the expected type for from_be_bytes we need to do it like this
    sig_bytes: [u8; (((SIG_BYTES * 8) + 7) / 8)],
    redc_param_bytes: [u8; SIG_BYTES + 1],
    exponent: u32,
    data_to_sign: [u8; DATA_TO_SIGN_MAX_LEN],
    data_to_sign_len: u32,
    pss_salt_len: u32,
) -> bool {
    assert(
        (SIG_BYTES == 512) | (SIG_BYTES == 256),
        "Only 2048-bit and 4096-bit RSA moduli are used in these passport tests",
    );
    assert(IS_PSS == 0, "These passport tests only use PKCS#1 v1.5 signatures");
    assert(HASH_BYTE_SIZE == 32, "These passport tests only use SHA-256 signatures");
    assert(pss_salt_len == 0, "PSS salt length must be zero for PKCS#1 v1.5");

    let pubkey =
        utils::pack_be_bytes_into_u128s::<SIG_BYTES, (SIG_BYTES + 14) / 15, 15>(pubkey_bytes);
    let redc_param = utils::pack_be_bytes_into_u128s::<SIG_BYTES + 1, _, 15>(redc_param_bytes);
    let params = BigNumParams::new(false, pubkey, redc_param);

    let signature = RuntimeBigNum::from_be_bytes(params, sig_bytes);
    let msg_hash = sha256_and_check_data_to_sign(data_to_sign, data_to_sign_len);
    verify_sha256_pkcs1v15::<_, SIG_BYTES * 8>(msg_hash, signature, exponent)
}
