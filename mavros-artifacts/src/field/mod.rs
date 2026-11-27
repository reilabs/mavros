//! Field elements and associated configuration for Mavros, allowing the field over which the
//! compiler and VM are running to be selected at runtime.

mod config;
mod element;
pub mod goldilocks;
pub mod limb_bits;
mod mavros_field;

/// The BN254 scalar field, implemented by arkworks' `Fr`.
pub type Bn254 = ark_bn254::Fr;

pub use config::{FieldConfig, IntoFieldElement};
pub use element::FieldElement;
pub use goldilocks::Goldilocks;
pub use limb_bits::LimbBits;
pub use mavros_field::{FieldId, MavrosField, UnknownField};
