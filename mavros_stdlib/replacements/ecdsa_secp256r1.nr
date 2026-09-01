use crate::mavros::replacements::ecdsa::verify_ecdsa;
use crate::mavros::vendor::bigcurve::curves::secp256r1::{
    SECP256r1_PARAM, Secp256r1,
};
use crate::mavros::vendor::bignum::{BigNum, Secp256r1_Fq, Secp256r1_Fr};

/// Pure-Noir implementation of the `ecdsa_secp256r1` low-level function.
pub fn ecdsa_secp256r1(
    public_key_x: [u8; 32],
    public_key_y: [u8; 32],
    signature: [u8; 64],
    message_hash: [u8; 32],
    predicate: bool,
) -> bool {
    let mut signature_r = [0; 32];
    let mut signature_s = [0; 32];
    for i in 0..32 {
        signature_r[i] = signature[i];
        signature_s[i] = signature[i + 32];
    }

    let half_order = Secp256r1_Fr::from_be_bytes([
        0x7f, 0xff, 0xff, 0xff, 0x80, 0x00, 0x00, 0x00,
        0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xde, 0x73, 0x7d, 0x56, 0xd3, 0x8b, 0xcf, 0x42,
        0x79, 0xdc, 0xe5, 0x61, 0x7e, 0x31, 0x92, 0xa8,
    ]);
    let safe_public_key_x = Secp256r1_Fq::from_be_bytes([
        85, 15, 71, 16, 3, 243, 223, 151,
        195, 223, 80, 106, 199, 151, 246, 114,
        31, 177, 161, 251, 123, 143, 111, 131,
        210, 36, 73, 138, 101, 200, 142, 36,
    ]);
    let safe_public_key_y = Secp256r1_Fq::from_be_bytes([
        19, 96, 147, 215, 1, 46, 80, 154,
        115, 113, 92, 189, 11, 0, 163, 204,
        15, 244, 181, 192, 27, 63, 250, 25,
        106, 177, 251, 50, 112, 54, 184, 230,
    ]);
    let safe_message_digest = Secp256r1_Fr::from_be_bytes([
        84, 112, 91, 163, 186, 175, 219, 223,
        186, 140, 95, 154, 112, 247, 168, 155,
        238, 152, 217, 6, 181, 62, 49, 7,
        77, 167, 186, 236, 220, 13, 169, 173,
    ]);
    let safe_r = Secp256r1_Fr::from_be_bytes([
        44, 112, 168, 208, 132, 182, 43, 252,
        92, 224, 54, 65, 202, 249, 247, 42,
        212, 218, 140, 129, 191, 230, 236, 148,
        135, 187, 94, 27, 239, 98, 161, 50,
    ]);
    let safe_s = Secp256r1_Fr::from_be_bytes([
        24, 173, 158, 226, 158, 175, 53, 31,
        220, 80, 241, 82, 12, 66, 94, 155,
        144, 138, 7, 39, 139, 67, 176, 236,
        123, 135, 39, 120, 193, 78, 7, 132,
    ]);

    verify_ecdsa::<Secp256r1_Fq, Secp256r1_Fr, Secp256r1>(
        Secp256r1_Fq::from_be_bytes(public_key_x),
        Secp256r1_Fq::from_be_bytes(public_key_y),
        Secp256r1_Fr::from_be_bytes(message_hash),
        Secp256r1_Fr::from_be_bytes(signature_r),
        Secp256r1_Fr::from_be_bytes(signature_s),
        predicate,
        SECP256r1_PARAM.a,
        SECP256r1_PARAM.b,
        half_order,
        safe_public_key_x,
        safe_public_key_y,
        safe_message_digest,
        safe_r,
        safe_s,
    )
}
