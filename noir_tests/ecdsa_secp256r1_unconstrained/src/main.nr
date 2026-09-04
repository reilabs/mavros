unconstrained fn main(
    hashed_message: [u8; 32],
    pub_key_x: [u8; 32],
    pub_key_y: [u8; 32],
    signature: [u8; 64],
) {
    // This positive vector was generated independently from the replacement's safety vector.
    assert(std::ecdsa_secp256r1::verify_signature(pub_key_x, pub_key_y, signature, hashed_message));

    // Verify the safety vector independently so reordered same-typed arguments cannot go unnoticed.
    let safe_pub_key_x = [
        85, 15, 71, 16, 3, 243, 223, 151, 195, 223, 80, 106, 199, 151, 246, 114, 31, 177, 161, 251,
        123, 143, 111, 131, 210, 36, 73, 138, 101, 200, 142, 36,
    ];
    let safe_pub_key_y = [
        19, 96, 147, 215, 1, 46, 80, 154, 115, 113, 92, 189, 11, 0, 163, 204, 15, 244, 181, 192, 27,
        63, 250, 25, 106, 177, 251, 50, 112, 54, 184, 230,
    ];
    let safe_signature = [
        44, 112, 168, 208, 132, 182, 43, 252, 92, 224, 54, 65, 202, 249, 247, 42, 212, 218, 140,
        129, 191, 230, 236, 148, 135, 187, 94, 27, 239, 98, 161, 50, 24, 173, 158, 226, 158, 175,
        53, 31, 220, 80, 241, 82, 12, 66, 94, 155, 144, 138, 7, 39, 139, 67, 176, 236, 123, 135, 39,
        120, 193, 78, 7, 132,
    ];
    let safe_hashed_message = [
        84, 112, 91, 163, 186, 175, 219, 223, 186, 140, 95, 154, 112, 247, 168, 155, 238, 152, 217,
        6, 181, 62, 49, 7, 77, 167, 186, 236, 220, 13, 169, 173,
    ];
    assert(std::ecdsa_secp256r1::verify_signature(
        safe_pub_key_x,
        safe_pub_key_y,
        safe_signature,
        safe_hashed_message,
    ));

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

    // The verification point for this signature has x = n + 3, which must not alias to r = 3.
    let alias_pub_key_x = [
        171, 131, 93, 152, 8, 208, 179, 233, 49, 153, 243, 141, 12, 28, 154, 90, 184, 201, 191, 98,
        81, 111, 251, 243, 123, 3, 126, 161, 127, 47, 189, 65,
    ];
    let alias_pub_key_y = [
        233, 156, 41, 51, 236, 91, 109, 233, 100, 9, 201, 197, 235, 230, 163, 56, 66, 88, 63, 22,
        128, 91, 150, 7, 78, 154, 151, 123, 18, 93, 41, 205,
    ];
    let alias_signature = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
