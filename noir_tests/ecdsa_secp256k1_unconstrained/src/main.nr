unconstrained fn main(
    hashed_message: [u8; 32],
    pub_key_x: [u8; 32],
    pub_key_y: [u8; 32],
    signature: [u8; 64],
) {
    assert(std::ecdsa_secp256k1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message));

    // The mathematically equivalent high-S signature is rejected to match Noir's BIP 62 rule.
    let high_s_signature = [
        229, 8, 28, 128, 171, 66, 125, 195, 112, 52, 111, 74, 14, 49, 170, 43,
        173, 141, 151, 152, 195, 128, 97, 219, 154, 229, 90, 78, 141, 244, 84, 253,
        215, 238, 103, 107, 203, 177, 142, 72, 120, 143, 51, 108, 226, 158, 11, 125,
        205, 243, 209, 92, 216, 93, 54, 210, 190, 112, 121, 242, 94, 214, 115, 236,
    ];
    assert(
        !std::ecdsa_secp256k1::verify_signature(
            pub_key_x,
            pub_key_y,
            high_s_signature,
            hashed_message,
        ),
    );

    let mut tampered_message = hashed_message;
    tampered_message[0] = tampered_message[0] ^ 1;
    assert(
        !std::ecdsa_secp256k1::verify_signature(pub_key_x, pub_key_y, signature, tampered_message),
    );
}
