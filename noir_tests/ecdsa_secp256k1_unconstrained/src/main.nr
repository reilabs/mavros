unconstrained fn main(
    hashed_message: [u8; 32],
    pub_key_x: [u8; 32],
    pub_key_y: [u8; 32],
    signature: [u8; 64],
) {
    // This positive vector was generated independently from the replacement's safety vector.
    assert(std::ecdsa_secp256k1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message));

    let high_s_signature = [
        89, 195, 87, 77, 86, 217, 195, 22, 87, 172, 213, 222, 78, 93, 186, 182, 1, 139, 240, 140,
        217, 76, 226, 113, 30, 17, 144, 100, 19, 249, 114, 245, 197, 114, 234, 1, 173, 109, 87, 27,
        188, 28, 223, 138, 106, 14, 241, 32, 27, 110, 216, 164, 232, 76, 50, 46, 216, 72, 240, 122,
        50, 126, 68, 14,
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

    let mut zero_r_signature = signature;
    let mut zero_s_signature = signature;
    let curve_order = [
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 186, 174,
        220, 230, 175, 72, 160, 59, 191, 210, 94, 140, 208, 54, 65, 65,
    ];
    let mut out_of_range_r_signature = signature;
    for i in 0..32 {
        zero_r_signature[i] = 0;
        zero_s_signature[i + 32] = 0;
        out_of_range_r_signature[i] = curve_order[i];
    }
    assert(
        !std::ecdsa_secp256k1::verify_signature(
            pub_key_x,
            pub_key_y,
            zero_r_signature,
            hashed_message,
        ),
    );
    assert(
        !std::ecdsa_secp256k1::verify_signature(
            pub_key_x,
            pub_key_y,
            zero_s_signature,
            hashed_message,
        ),
    );
    assert(
        !std::ecdsa_secp256k1::verify_signature(
            pub_key_x,
            pub_key_y,
            out_of_range_r_signature,
            hashed_message,
        ),
    );

    let mut off_curve_y = pub_key_y;
    off_curve_y[31] = off_curve_y[31] ^ 1;
    assert(
        !std::ecdsa_secp256k1::verify_signature(pub_key_x, off_curve_y, signature, hashed_message),
    );
    let field_modulus = [
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 252, 47,
    ];
    assert(
        !std::ecdsa_secp256k1::verify_signature(
            field_modulus,
            pub_key_y,
            signature,
            hashed_message,
        ),
    );
    assert(!std::ecdsa_secp256k1::verify_signature([0; 32], [0; 32], [0; 64], [0; 32]));

    // The verification point for this signature has x = n + 2, which must not alias to r = 2.
    let alias_pub_key_x = [
        25, 9, 159, 189, 221, 205, 183, 161, 68, 16, 181, 198, 186, 115, 245, 27, 210, 247, 74, 61,
        28, 216, 69, 111, 204, 29, 166, 142, 214, 70, 136, 132,
    ];
    let alias_pub_key_y = [
        176, 123, 159, 125, 217, 234, 87, 83, 224, 9, 36, 106, 29, 184, 94, 126, 22, 68, 191, 222,
        226, 118, 136, 26, 149, 12, 229, 159, 221, 69, 180, 91,
    ];
    let alias_signature = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ];
    let alias_message = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1,
    ];
    assert(
        !std::ecdsa_secp256k1::verify_signature(
            alias_pub_key_x,
            alias_pub_key_y,
            alias_signature,
            alias_message,
        ),
    );

    assert(std::mavros::replacements::ecdsa_secp256k1::ecdsa_secp256k1(
        [0; 32],
        [0; 32],
        [0; 64],
        [0; 32],
        false,
    ));
}
