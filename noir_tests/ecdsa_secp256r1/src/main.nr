// End-to-end check of the circuit-backed `ecdsa_secp256r1` replacement on well-formed inputs.
// The test vector is a deterministic low-s P-256 signature generated from fixed `d` and `k`.
fn main(pub_x: [u8; 32], pub_y: [u8; 32], signature: [u8; 64], hash: [u8; 32]) {
    // A valid signature verifies.
    assert(std::ecdsa_secp256r1::verify_signature(pub_x, pub_y, signature, hash));

    // The same signature over a tampered hash is rejected without trapping.
    let mut tampered_hash = hash;
    tampered_hash[0] ^= 1;
    assert(!std::ecdsa_secp256r1::verify_signature(pub_x, pub_y, signature, tampered_hash));
}
