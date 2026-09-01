fn main(hashed_message: [u8; 32], pub_key_x: [u8; 32], pub_key_y: [u8; 32], signature: [u8; 64]) {
    // This positive vector was generated independently from the replacement's safety vector.
    assert(std::ecdsa_secp256k1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message));
}
