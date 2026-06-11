//! Pure-Noir implementations of black-box functions that play the role of builtins in upstream
//! Noir. The mavros driver rewrites the corresponding `#[foreign]` stdlib shims after parsing to
//! call into this module (see `FOREIGN_REPLACEMENTS` in the compiler's `project.rs`), so the rest
//! of the frontend treats them as ordinary functions.

pub mod blake3;
pub mod poseidon2_permutation;
pub mod sha256_compression;
