fn main(hashed_message: [u8; 32], pub_key_x: [u8; 32], pub_key_y: [u8; 32], signature: [u8; 64]) {
    // The verification point has x = n + 3. Reducing x into Fr would alias it to r = 3 and
    // incorrectly accept the signature; ACVM rejects this noncanonical x-coordinate.
    assert(
        !std::ecdsa_secp256r1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message),
    );
}
