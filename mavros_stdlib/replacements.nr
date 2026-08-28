//! Pure-Noir implementations of black-box functions that play the role of builtins in upstream
//! Noir. The mavros driver rewrites the corresponding stdlib shims after parsing to call into this
//! module (see the replacement registries in the compiler's `project.rs`), so the rest of the
//! frontend treats them as ordinary functions.

pub mod blake3;
pub mod derive_pedersen_generators;
pub mod embedded_curve_add;
pub mod multi_scalar_mul;
pub mod poseidon2_permutation;
pub mod sha256_compression;
