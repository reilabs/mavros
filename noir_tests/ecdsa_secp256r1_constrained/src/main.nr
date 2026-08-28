fn main(hashed_message: [u8; 32], pub_key_x: [u8; 32], pub_key_y: [u8; 32], signature: [u8; 64]) {
    assert(std::ecdsa_secp256r1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message));

    let mut tampered_message = hashed_message;
    tampered_message[0] = tampered_message[0] ^ 1;
    assert(
        !std::ecdsa_secp256r1::verify_signature(pub_key_x, pub_key_y, signature, tampered_message),
    );
}
