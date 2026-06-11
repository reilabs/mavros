//! Customized collections for the compiler.
//!
//! # HashMap and HashSet
//!
//! `std`'s default [`RandomState`](std::collections::hash_map::RandomState) seeds every map
//! differently per process, making iteration order nondeterministic. The compilation pipeline
//! allocates identifiers from monotonic counters (e.g. `SSA::fresh_value`) *while iterating* such
//! maps, so random iteration order leaks into emitted identifiers and ultimately into every dump
//! and artifact. Since the pipeline is single-threaded, fixing the hasher makes compilation a pure
//! function of its input.
//!
//! All compiler code must use these aliases instead of `std::collections::{HashMap, HashSet}`; this
//! is enforced by `disallowed-types` in the workspace `clippy.toml`.
//!
//! Notes:
//!
//! - Construct with `HashMap::default()`, not `::new()` — the inherent `new` only exists for
//!   the `RandomState` hasher.
//! - FxHash output depends on the target word size, so iteration order may differ between 32- and
//!   64-bit hosts. Run-to-run determinism on one host is the goal here; cross-architecture
//!   reproducibility is out of scope.

pub use rustc_hash::FxBuildHasher;

// HASHED COLLECTIONS
// ================================================================================================

#[allow(clippy::disallowed_types)]
pub type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

#[allow(clippy::disallowed_types)]
pub type HashSet<T> = std::collections::HashSet<T, FxBuildHasher>;
