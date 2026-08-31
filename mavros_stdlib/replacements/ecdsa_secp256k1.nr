use crate::mavros::vendor::ecdsa_secp256k1::bigcurve::curves::secp256k1::{
    SECP256k1_PARAM, Secp256k1,
};
use crate::mavros::vendor::ecdsa_secp256k1::bignum::{
    Secp256k1_Fq, Secp256k1_Fr,
};
use crate::mavros::vendor::ecdsa_secp256r1::bigcurve::{BigCurve, ScalarField};
use crate::mavros::vendor::ecdsa_secp256r1::bignum::BigNum;
use crate::mavros::vendor::ecdsa_secp256r1::bignum::bignum::conditional_select;

fn verify(
    public_key_x: Secp256k1_Fq,
    public_key_y: Secp256k1_Fq,
    message_digest: [u8; 32],
    r: Secp256k1_Fr,
    s: Secp256k1_Fr,
) -> bool {
    let mut formatted_message_digest = Secp256k1_Fr::zero().to_le_bytes();
    for i in 0..formatted_message_digest.len() {
        formatted_message_digest[i] = message_digest[i];
    }
    let e = Secp256k1_Fr::from_be_bytes(formatted_message_digest);

    let w = unsafe { s.__invmod() };
    assert(s * w == Secp256k1_Fr::one());
    let u_1: ScalarField<65> = ScalarField::from_bignum(e * w);
    let u_2: ScalarField<65> = ScalarField::from_bignum(r * w);

    // The caller validates and sanitizes the coordinates before entering the verifier.
    let public_key = Secp256k1::from_coordinates_unsafe(public_key_x, public_key_y, false);
    let result = Secp256k1::evaluate_linear_expression(
        [Secp256k1::one(), public_key],
        [u_1, u_2],
        [],
    );

    let mut result_x = Secp256k1_Fr::zero();
    for i in 0..Secp256k1_Fr::num_limbs(Secp256k1_Fr::zero()) {
        result_x.set_limb(i, result.x().get_limb(i));
    }
    result_x == r
}

/// Pure-Noir implementation of the `ecdsa_secp256k1` low-level function.
pub fn ecdsa_secp256k1(
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

    let public_key_x = Secp256k1_Fq::from_be_bytes(public_key_x);
    let public_key_y = Secp256k1_Fq::from_be_bytes(public_key_y);
    let signature_r = Secp256k1_Fr::from_be_bytes(signature_r);
    let signature_s = Secp256k1_Fr::from_be_bytes(signature_s);

    // The native black box is total for malformed inputs and when its predicate is false. The
    // bignum gadget uses assertions internally, so substitute one known-good vector before
    // entering it and mask the result afterwards.
    let safe_public_key_x = Secp256k1_Fq::from_be_bytes([
        0xa0, 0x43, 0x4d, 0x9e, 0x47, 0xf3, 0xc8, 0x62,
        0x35, 0x47, 0x7c, 0x7b, 0x1a, 0xe6, 0xae, 0x5d,
        0x34, 0x42, 0xd4, 0x9b, 0x19, 0x43, 0xc2, 0xb7,
        0x52, 0xa6, 0x8e, 0x2a, 0x47, 0xe2, 0x47, 0xc7,
    ]);
    let safe_public_key_y = Secp256k1_Fq::from_be_bytes([
        0x89, 0x3a, 0xba, 0x42, 0x54, 0x19, 0xbc, 0x27,
        0xa3, 0xb6, 0xc7, 0xe6, 0x93, 0xa2, 0x4c, 0x69,
        0x6f, 0x79, 0x4c, 0x2e, 0xd8, 0x77, 0xa1, 0x59,
        0x3c, 0xbe, 0xe5, 0x3b, 0x03, 0x73, 0x68, 0xd7,
    ]);
    // BigNum arithmetic assumes canonical limbs. Check the untrusted coordinates first and use a
    // known curve point for the curve equation when either coordinate is out of range. The range
    // bit is retained in `point_valid`, so sanitizing cannot turn an invalid key into a valid one.
    let point_in_range = (public_key_x < Secp256k1_Fq::modulus())
        & (public_key_y < Secp256k1_Fq::modulus());
    let checked_public_key_x =
        conditional_select(public_key_x, safe_public_key_x, point_in_range);
    let checked_public_key_y =
        conditional_select(public_key_y, safe_public_key_y, point_in_range);
    let point_on_curve = checked_public_key_y * checked_public_key_y
        == (checked_public_key_x * checked_public_key_x + SECP256k1_PARAM.a)
                * checked_public_key_x
            + SECP256k1_PARAM.b;
    let point_valid = point_in_range & point_on_curve;

    // Noir's native secp256k1 verifier rejects malleable high-S signatures (BIP 62).
    let half_order = Secp256k1_Fr::from_be_bytes([
        0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x5d, 0x57, 0x6e, 0x73, 0x57, 0xa4, 0x50, 0x1d,
        0xdf, 0xe9, 0x2f, 0x46, 0x68, 0x1b, 0x20, 0xa0,
    ]);
    let signature_valid = (signature_r != Secp256k1_Fr::zero())
        & (signature_r < Secp256k1_Fr::modulus())
        & (signature_s != Secp256k1_Fr::zero())
        & (signature_s <= half_order);
    let safe_signature_r = Secp256k1_Fr::from_be_bytes([
        0xe5, 0x08, 0x1c, 0x80, 0xab, 0x42, 0x7d, 0xc3,
        0x70, 0x34, 0x6f, 0x4a, 0x0e, 0x31, 0xaa, 0x2b,
        0xad, 0x8d, 0x97, 0x98, 0xc3, 0x80, 0x61, 0xdb,
        0x9a, 0xe5, 0x5a, 0x4e, 0x8d, 0xf4, 0x54, 0xfd,
    ]);
    let safe_signature_s = Secp256k1_Fr::from_be_bytes([
        0x28, 0x11, 0x98, 0x94, 0x34, 0x4e, 0x71, 0xb7,
        0x87, 0x70, 0xcc, 0x93, 0x1d, 0x61, 0xf4, 0x80,
        0xec, 0xbb, 0x0b, 0x89, 0xd6, 0xeb, 0x69, 0x69,
        0x01, 0x61, 0xe4, 0x9a, 0x71, 0x5f, 0xcd, 0x55,
    ]);
    let inputs_valid = predicate & point_valid & signature_valid;
    let public_key_x = conditional_select(public_key_x, safe_public_key_x, inputs_valid);
    let public_key_y = conditional_select(public_key_y, safe_public_key_y, inputs_valid);
    let signature_r = conditional_select(signature_r, safe_signature_r, inputs_valid);
    let signature_s = conditional_select(signature_s, safe_signature_s, inputs_valid);
    let safe_message_hash = [
        0x3a, 0x73, 0xf4, 0x12, 0x3a, 0x5c, 0xd2, 0x12,
        0x1f, 0x21, 0xcd, 0x7e, 0x8d, 0x35, 0x88, 0x35,
        0x47, 0x69, 0x49, 0xd0, 0x35, 0xd9, 0xc2, 0xda,
        0x68, 0x06, 0xb4, 0x63, 0x3a, 0xc8, 0xc1, 0xe2,
    ];
    let mut checked_message_hash = [0; 32];
    for i in 0..32 {
        checked_message_hash[i] = if inputs_valid { message_hash[i] } else { safe_message_hash[i] };
    }

    let verified = verify(
        public_key_x,
        public_key_y,
        checked_message_hash,
        signature_r,
        signature_s,
    );
    inputs_valid & verified
}
