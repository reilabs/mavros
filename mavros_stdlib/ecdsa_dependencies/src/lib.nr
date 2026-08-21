use bignum::bignum::conditional_select;
use bignum::{BigNum, Secp256r1_Fq, Secp256r1_Fr};
use ecdsa::ecdsa::verify_secp256r1_ecdsa;

/// Circuit implementation used in place of Noir's `ecdsa_secp256r1` black box.
pub fn verify_signature(
    public_key_x: [u8; 32],
    public_key_y: [u8; 32],
    signature: [u8; 64],
    message_hash: [u8; 32],
) -> bool {
    let mut signature_r = [0; 32];
    let mut signature_s = [0; 32];
    for i in 0..32 {
        signature_r[i] = signature[i];
        signature_s[i] = signature[i + 32];
    }

    let public_key_x = Secp256r1_Fq::from_be_bytes(public_key_x);
    let public_key_y = Secp256r1_Fq::from_be_bytes(public_key_y);
    let signature_r = Secp256r1_Fr::from_be_bytes(signature_r);
    let signature_s = Secp256r1_Fr::from_be_bytes(signature_s);

    // The native black box is total when its surrounding branch predicate is false. Mavros
    // lowers this implementation as an ordinary function, so sanitize malformed inputs before
    // entering the assertion-based big-curve gadget and mask its result afterwards.
    let curve_a = Secp256r1_Fq::from_be_bytes([
        0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfc,
    ]);
    let curve_b = Secp256r1_Fq::from_be_bytes([
        0x5a, 0xc6, 0x35, 0xd8, 0xaa, 0x3a, 0x93, 0xe7,
        0xb3, 0xeb, 0xbd, 0x55, 0x76, 0x98, 0x86, 0xbc,
        0x65, 0x1d, 0x06, 0xb0, 0xcc, 0x53, 0xb0, 0xf6,
        0x3b, 0xce, 0x3c, 0x3e, 0x27, 0xd2, 0x60, 0x4b,
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
    // BigNum arithmetic assumes canonical limbs. Check the untrusted coordinates first and use a
    // known curve point for the curve equation when either coordinate is out of range. The range
    // bit is retained in `point_valid`, so sanitizing cannot turn an invalid key into a valid one.
    let point_in_range = (public_key_x < Secp256r1_Fq::modulus())
        & (public_key_y < Secp256r1_Fq::modulus());
    let checked_public_key_x =
        conditional_select(public_key_x, safe_public_key_x, point_in_range);
    let checked_public_key_y =
        conditional_select(public_key_y, safe_public_key_y, point_in_range);
    let point_on_curve = checked_public_key_y * checked_public_key_y
        == (checked_public_key_x * checked_public_key_x + curve_a) * checked_public_key_x + curve_b;
    let point_valid = point_in_range & point_on_curve;

    let half_order = Secp256r1_Fr::from_be_bytes([
        0x7f, 0xff, 0xff, 0xff, 0x80, 0x00, 0x00, 0x00,
        0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xde, 0x73, 0x7d, 0x56, 0xd3, 0x8b, 0xcf, 0x42,
        0x79, 0xdc, 0xe5, 0x61, 0x7e, 0x31, 0x92, 0xa8,
    ]);
    let signature_valid = (signature_r != Secp256r1_Fr::zero())
        & (signature_r < Secp256r1_Fr::modulus())
        & (signature_s != Secp256r1_Fr::zero())
        & (signature_s <= half_order);
    let safe_signature_r = Secp256r1_Fr::from_be_bytes([
        44, 112, 168, 208, 132, 182, 43, 252,
        92, 224, 54, 65, 202, 249, 247, 42,
        212, 218, 140, 129, 191, 230, 236, 148,
        135, 187, 94, 27, 239, 98, 161, 50,
    ]);
    let safe_signature_s = Secp256r1_Fr::from_be_bytes([
        24, 173, 158, 226, 158, 175, 53, 31,
        220, 80, 241, 82, 12, 66, 94, 155,
        144, 138, 7, 39, 139, 67, 176, 236,
        123, 135, 39, 120, 193, 78, 7, 132,
    ]);
    let inputs_valid = point_valid & signature_valid;
    let public_key_x = conditional_select(public_key_x, safe_public_key_x, inputs_valid);
    let public_key_y = conditional_select(public_key_y, safe_public_key_y, inputs_valid);
    let signature_r = conditional_select(signature_r, safe_signature_r, inputs_valid);
    let signature_s = conditional_select(signature_s, safe_signature_s, inputs_valid);
    let safe_message_hash = [
        84, 112, 91, 163, 186, 175, 219, 223,
        186, 140, 95, 154, 112, 247, 168, 155,
        238, 152, 217, 6, 181, 62, 49, 7,
        77, 167, 186, 236, 220, 13, 169, 173,
    ];
    let mut checked_message_hash = [0; 32];
    for i in 0..32 {
        checked_message_hash[i] = if inputs_valid { message_hash[i] } else { safe_message_hash[i] };
    }

    let verified = verify_secp256r1_ecdsa(
        public_key_x,
        public_key_y,
        checked_message_hash,
        (signature_r, signature_s),
    );
    inputs_valid & verified
}
