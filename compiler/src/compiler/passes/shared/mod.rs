//! Functionality shared between multiple passes.

pub mod availability;
pub mod divmod_guard;
pub mod field_windows;
pub mod limbs;
pub mod overflow_guard;
pub mod rewrite_asserts;
pub mod seq_bounds;
pub mod shift_guard;
pub mod unreachable;
pub mod value_replacements;

/// The shortest run of identical steps, such as range checks or return bindings, that a lowering
/// emits as a counted loop.
pub const MIN_LOOPED_RUN: usize = 4;
