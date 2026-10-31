//! Pure-Noir implementations of black-box functions that play the role of builtins in upstream
//! Noir, plus shared support code used by those implementations. The mavros driver rewrites the
//! corresponding stdlib shims after parsing to call into this module (see the replacement
//! registries in the compiler's `project.rs`), so the rest of the frontend treats them as ordinary
//! functions.

pub mod aes128_encrypt;
pub mod blake2s;
pub mod blake3;
pub mod blake_helper;
// These replacements depend on BN254's embedded curve or large-field arithmetic.
#[field(bn254)]
pub mod derive_pedersen_generators;
#[field(bn254)]
pub mod ecdsa;
#[field(bn254)]
pub mod ecdsa_secp256k1;
#[field(bn254)]
pub mod ecdsa_secp256r1;
#[field(bn254)]
pub mod embedded_curve_add;
#[field(bn254)]
pub mod multi_scalar_mul;
#[field(bn254)]
pub mod poseidon2_permutation;
pub mod sha256_compression;
