// Malformed ECDSA inputs must be rejected as `false`, never trap witness generation: the
// circuit replacement sanitizes its inputs before the assertion-based bignum/bigcurve gadgets.
fn main(pub_x: [u8; 32], pub_y: [u8; 32], signature: [u8; 64], hash: [u8; 32], take_branch: bool) {
    // A zero `s` with an otherwise valid key: exercises the signature-validity mask alone.
    let mut zero_s = signature;
    for i in 0..32 {
        zero_s[i + 32] = 0;
    }
    assert(!std::ecdsa_secp256r1::verify_signature(pub_x, pub_y, zero_s, hash));

    // An out-of-range public key inside an inactive witness branch: mavros executes both sides
    // of a witness conditional, so this call runs on garbage and must still stay total.
    let bad_x = [255; 32];
    if take_branch {
        assert(std::ecdsa_secp256r1::verify_signature(bad_x, bad_x, zero_s, hash));
    }
}
