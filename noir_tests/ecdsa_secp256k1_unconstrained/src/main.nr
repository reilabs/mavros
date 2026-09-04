unconstrained fn main(
    hashed_message: [u8; 32],
    pub_key_x: [u8; 32],
    pub_key_y: [u8; 32],
    signature: [u8; 64],
) {
    // This positive vector was generated independently from the replacement's safety vector.
    assert(std::ecdsa_secp256k1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message));

    // Verify the safety vector independently so reordered same-typed arguments cannot go unnoticed.
    let safe_pub_key_x = [
        0xa0, 0x43, 0x4d, 0x9e, 0x47, 0xf3, 0xc8, 0x62, 0x35, 0x47, 0x7c, 0x7b, 0x1a, 0xe6, 0xae,
        0x5d, 0x34, 0x42, 0xd4, 0x9b, 0x19, 0x43, 0xc2, 0xb7, 0x52, 0xa6, 0x8e, 0x2a, 0x47, 0xe2,
        0x47, 0xc7,
    ];
    let safe_pub_key_y = [
        0x89, 0x3a, 0xba, 0x42, 0x54, 0x19, 0xbc, 0x27, 0xa3, 0xb6, 0xc7, 0xe6, 0x93, 0xa2, 0x4c,
        0x69, 0x6f, 0x79, 0x4c, 0x2e, 0xd8, 0x77, 0xa1, 0x59, 0x3c, 0xbe, 0xe5, 0x3b, 0x03, 0x73,
        0x68, 0xd7,
    ];
    let safe_signature = [
        0xe5, 0x08, 0x1c, 0x80, 0xab, 0x42, 0x7d, 0xc3, 0x70, 0x34, 0x6f, 0x4a, 0x0e, 0x31, 0xaa,
        0x2b, 0xad, 0x8d, 0x97, 0x98, 0xc3, 0x80, 0x61, 0xdb, 0x9a, 0xe5, 0x5a, 0x4e, 0x8d, 0xf4,
        0x54, 0xfd, 0x28, 0x11, 0x98, 0x94, 0x34, 0x4e, 0x71, 0xb7, 0x87, 0x70, 0xcc, 0x93, 0x1d,
        0x61, 0xf4, 0x80, 0xec, 0xbb, 0x0b, 0x89, 0xd6, 0xeb, 0x69, 0x69, 0x01, 0x61, 0xe4, 0x9a,
        0x71, 0x5f, 0xcd, 0x55,
    ];
    let safe_hashed_message = [
        0x3a, 0x73, 0xf4, 0x12, 0x3a, 0x5c, 0xd2, 0x12, 0x1f, 0x21, 0xcd, 0x7e, 0x8d, 0x35, 0x88,
        0x35, 0x47, 0x69, 0x49, 0xd0, 0x35, 0xd9, 0xc2, 0xda, 0x68, 0x06, 0xb4, 0x63, 0x3a, 0xc8,
        0xc1, 0xe2,
    ];
    assert(std::ecdsa_secp256k1::verify_signature(
        safe_pub_key_x,
        safe_pub_key_y,
        safe_signature,
        safe_hashed_message,
    ));

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
