use crate::mavros::vendor::bigcurve::{BigCurve, ScalarField};
use crate::mavros::vendor::bignum::BigNum;
use crate::mavros::vendor::bignum::bignum::conditional_select;

fn verify<Fq, Fr, Curve, let N: u32>(
    public_key_x: Fq,
    public_key_y: Fq,
    message_digest: Fr,
    r: Fr,
    s: Fr,
) -> bool
where
    Fq: BigNum<N = N, MOD_BITS = 256>,
    Fr: BigNum<N = N, MOD_BITS = 256>,
    Fr: crate::cmp::Ord,
    Curve: BigCurve<Fq>,
{
    let w = unsafe { s.__invmod() };
    assert(s * w == Fr::one());
    // `ScalarField::from_bignum` consumes 5 slices from the 16-bit high limb of a 256-bit value,
    // then 30 slices for every remaining 120-bit limb. The curve parameters must correspondingly
    // satisfy offset_generator_final = offset_generator * 2^(4 * (NScalarSlices - 1)).
    let u_1: ScalarField<5 + 30 * (N - 1)> =
        ScalarField::from_bignum(message_digest * w);
    let u_2: ScalarField<5 + 30 * (N - 1)> =
        ScalarField::from_bignum(r * w);

    // The caller validates and sanitizes the coordinates before entering the verifier.
    let public_key = Curve::from_coordinates_unsafe(public_key_x, public_key_y, false);
    let result = Curve::evaluate_linear_expression(
        [Curve::one(), public_key],
        [u_1, u_2],
        [],
    );

    // Match the native ACVM verifier: an x-coordinate outside the scalar field is invalid. Merely
    // comparing it with `r` as an Fr would reduce it modulo the curve order and accept aliases.
    let mut result_x = Fr::zero();
    for i in 0..Fr::num_limbs(Fr::zero()) {
        result_x.set_limb(i, result.x().get_limb(i));
    }
    (result_x < Fr::modulus()) & (result_x == r)
}

/// Shared ECDSA verification flow for the secp256k1 and secp256r1 replacements.
pub fn verify_ecdsa<Fq, Fr, Curve, let N: u32>(
    public_key_x: Fq,
    public_key_y: Fq,
    message_digest: Fr,
    r: Fr,
    s: Fr,
    predicate: bool,
    curve_b: Fq,
    half_order: Fr,
    safe_public_key_x: Fq,
    safe_public_key_y: Fq,
    safe_message_digest: Fr,
    safe_r: Fr,
    safe_s: Fr,
) -> bool
where
    // Sharing N in both bounds makes mismatched base/scalar limb counts a type error.
    Fq: BigNum<N = N, MOD_BITS = 256>,
    Fq: crate::cmp::Ord,
    Fr: BigNum<N = N, MOD_BITS = 256>,
    Fr: crate::cmp::Ord,
    Curve: BigCurve<Fq>,
{
    // BigNum arithmetic assumes canonical limbs. Check the untrusted coordinates first and use a
    // known curve point for the curve equation when either coordinate is out of range. The range
    // bit is retained in `point_valid`, so sanitizing cannot make an invalid key valid.
    let point_in_range =
        (public_key_x < Fq::modulus()) & (public_key_y < Fq::modulus());
    let checked_public_key_x =
        conditional_select(public_key_x, safe_public_key_x, point_in_range);
    let checked_public_key_y =
        conditional_select(public_key_y, safe_public_key_y, point_in_range);
    let point_on_curve = checked_public_key_y * checked_public_key_y
        == (checked_public_key_x * checked_public_key_x + Curve::a())
                * checked_public_key_x
            + curve_b;
    let point_valid = point_in_range & point_on_curve;

    let signature_valid = (r != Fr::zero())
        & (r < Fr::modulus())
        & (s != Fr::zero())
        & (s <= half_order);
    let inputs_valid = predicate & point_valid & signature_valid;

    // The bignum and curve gadgets assert on malformed inputs. Substitute one known-good vector
    // before entering them, and mask that vector's result afterwards.
    let public_key_x = conditional_select(public_key_x, safe_public_key_x, inputs_valid);
    let public_key_y = conditional_select(public_key_y, safe_public_key_y, inputs_valid);
    let message_digest = conditional_select(message_digest, safe_message_digest, inputs_valid);
    let r = conditional_select(r, safe_r, inputs_valid);
    let s = conditional_select(s, safe_s, inputs_valid);
    let verified = verify::<Fq, Fr, Curve, N>(public_key_x, public_key_y, message_digest, r, s);

    // ACVM treats a disabled black-box call as successful, irrespective of its input values.
    !predicate | (inputs_valid & verified)
}
