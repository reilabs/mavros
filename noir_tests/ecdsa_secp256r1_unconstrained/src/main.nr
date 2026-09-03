unconstrained fn main(
    hashed_message: [u8; 32],
    pub_key_x: [u8; 32],
    pub_key_y: [u8; 32],
    signature: [u8; 64],
) {
    // This positive vector was generated independently from the replacement's safety vector.
    assert(std::ecdsa_secp256r1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message));

    let high_s_signature = [
        116, 99, 98, 91, 122, 80, 126, 41, 141, 26, 213, 42, 133, 82, 60, 115, 35, 6, 107, 160, 147,
        69, 252, 153, 117, 204, 158, 251, 80, 47, 34, 73, 204, 183, 47, 67, 166, 157, 9, 240, 53,
        102, 51, 55, 79, 224, 72, 113, 48, 131, 243, 202, 165, 76, 13, 165, 166, 75, 215, 52, 0,
        203, 71, 46,
    ];
    assert(
        !std::ecdsa_secp256r1::verify_signature(
            pub_key_x,
            pub_key_y,
            high_s_signature,
            hashed_message,
        ),
    );

    let mut tampered_message = hashed_message;
    tampered_message[0] = tampered_message[0] ^ 1;
    assert(
        !std::ecdsa_secp256r1::verify_signature(pub_key_x, pub_key_y, signature, tampered_message),
    );

    let mut zero_r_signature = signature;
    let mut zero_s_signature = signature;
    let curve_order = [
        255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 188, 230, 250, 173,
        167, 23, 158, 132, 243, 185, 202, 194, 252, 99, 37, 81,
    ];
    let mut out_of_range_r_signature = signature;
    for i in 0..32 {
        zero_r_signature[i] = 0;
        zero_s_signature[i + 32] = 0;
        out_of_range_r_signature[i] = curve_order[i];
    }
    assert(
        !std::ecdsa_secp256r1::verify_signature(
            pub_key_x,
            pub_key_y,
            zero_r_signature,
            hashed_message,
        ),
    );
    assert(
        !std::ecdsa_secp256r1::verify_signature(
            pub_key_x,
            pub_key_y,
            zero_s_signature,
            hashed_message,
        ),
    );
    assert(
        !std::ecdsa_secp256r1::verify_signature(
            pub_key_x,
            pub_key_y,
            out_of_range_r_signature,
            hashed_message,
        ),
    );

    let mut off_curve_y = pub_key_y;
    off_curve_y[31] = off_curve_y[31] ^ 1;
    assert(
        !std::ecdsa_secp256r1::verify_signature(pub_key_x, off_curve_y, signature, hashed_message),
    );
    let field_modulus = [
        255, 255, 255, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255,
    ];
    assert(
        !std::ecdsa_secp256r1::verify_signature(
            field_modulus,
            pub_key_y,
            signature,
            hashed_message,
        ),
    );
    assert(!std::ecdsa_secp256r1::verify_signature([0; 32], [0; 32], [0; 64], [0; 32]));

    // The verification point for this signature has x = n + 1, which must not alias to r = 1.
    let alias_pub_key_x = [
        251, 1, 10, 155, 99, 72, 96, 71, 99, 109, 145, 204, 236, 47, 228, 0, 110, 40, 0, 225, 51,
        78, 118, 68, 58, 110, 11, 234, 42, 117, 227, 173,
    ];
    let alias_pub_key_y = [
        66, 173, 71, 65, 248, 239, 109, 119, 150, 39, 61, 34, 97, 49, 144, 116, 246, 250, 50, 195,
        100, 166, 193, 141, 81, 121, 93, 235, 27, 254, 161, 182,
    ];
    let alias_signature = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ];
    let alias_message = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1,
    ];
    assert(
        !std::ecdsa_secp256r1::verify_signature(
            alias_pub_key_x,
            alias_pub_key_y,
            alias_signature,
            alias_message,
        ),
    );

    // A false black-box predicate is successful even when every supplied value is malformed.
    assert(std::mavros::replacements::ecdsa_secp256r1::ecdsa_secp256r1(
        [0; 32],
        [0; 32],
        [0; 64],
        [0; 32],
        false,
    ));
}
