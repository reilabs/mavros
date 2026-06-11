//! TEMPORARY: source-field bridge for the Goldilocks port.
//!
//! Mavros's constraint field is hardcoded to bn254 (`mavros-artifacts`). Under a Goldilocks Noir
//! build, Noir's field element is a different `ark_ff` type, so the usual `into_repr()` does not
//! type-check against Mavros's `Field`; this re-decodes via canonical little-endian bytes instead.
//!
//! Remove this module once `mavros-artifacts` is field-generic — its two call sites
//! (`compiler::lowering::expression_converter` and `abi_helpers`) then convert through the
//! artifact's native field type directly. Kept in one place precisely so that removal is a single
//! deletion rather than a hunt for inlined copies.

use mavros_artifacts::Field;

/// Convert a Noir field element's ark representation (from `FieldElement::into_repr()`) into
/// Mavros's bn254 constraint field, via canonical little-endian bytes. Field-agnostic: works
/// whether Noir was built with bn254 or a smaller field. Plain `into_repr()` only type-checks when
/// both are the same ark field.
pub fn noir_field_to_bn254<F: ark_ff::PrimeField>(repr: F) -> Field {
    use ark_ff::{BigInteger, PrimeField};
    Field::from_le_bytes_mod_order(&repr.into_bigint().to_bytes_le())
}
