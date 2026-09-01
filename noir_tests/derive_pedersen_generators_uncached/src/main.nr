use std::embedded_curve_ops::EmbeddedCurvePoint;
use std::hash::derive_generators;

fn main() {
    // A non-standard domain bypasses both fast paths and exercises BLAKE3-to-field reduction,
    // Tonelli-Shanks square roots, and try-and-increment hash-to-curve. This vector was generated
    // independently with the reference `bn254_blackbox_solver` implementation.
    let generator: [EmbeddedCurvePoint; 1] = derive_generators("test domain".as_bytes(), 0);
    assert_eq(
        generator[0].x,
        0x12745f0888c51339012f08867d18b5135a65cbe65cc71c0e2808830c2c600806,
    );
    assert_eq(
        generator[0].y,
        0x21f71fa8aa246d40ac6018739828a529e83bc3d5d6c158779a6d07fded1ce254,
    );
}
