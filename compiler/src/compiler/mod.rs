pub mod analysis;
pub mod codegen;
pub mod lowering;
pub mod pass_manager;
pub mod passes;
pub mod ssa;
pub mod untaint_control_flow;
pub mod util;

pub use mavros_artifacts::Field;

/// Convert a Noir field element's ark representation (from `FieldElement::into_repr()`)
/// into Mavros's bn254 constraint field, via canonical little-endian bytes. Field-agnostic:
/// works whether Noir was built with bn254 or a smaller field (e.g. Goldilocks), so it
/// decouples Mavros's `Field` from Noir's source field type. Plain `into_repr()` only
/// type-checks when both are the same ark field.
pub fn noir_field_to_bn254<F: ark_ff::PrimeField>(repr: F) -> Field {
    use ark_ff::{BigInteger, PrimeField};
    Field::from_le_bytes_mod_order(&repr.into_bigint().to_bytes_le())
}
