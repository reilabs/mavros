//! Configuration for the field over which the program is running.
//!
//! [`FieldConfig`] carries the field's modulus, bit-width, and the derived constants the compiler
//! mints values from (e.g. `two_pow`). This module is a **stub**: it is defined and unit-tested,
//! but nothing threads it yet. Phase 3 will home a `FieldConfig` on the `SSA` and route the
//! modulus/width/`two_pow` sites through it; today those sites still read the constants directly.

use ark_ff::BigInt;

use crate::field::{element::FieldElement, mavros_field::FieldId};

// RUNTIME FIELD CONFIGURATION
// ================================================================================================

/// The configuration of the field a program operates over.
///
/// bn254 is the only field today, so this is effectively a tag; it exists so that Phase 3 has a
/// single object to thread through the SSA and the symbolic evaluators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldConfig {
    id: FieldId,
}

// The constant accessors below are **bn254-pinned** as they ignore `self.id` and read
// `FieldElement`'s constants directly.
//
// Phase 3 threads real per-field data through here; until then the `debug_assert!`s guard against a
// second `FieldId` silently receiving bn254's values.
impl FieldConfig {
    /// The bn254 configuration.
    #[must_use]
    pub const fn bn254() -> FieldConfig {
        FieldConfig { id: FieldId::Bn254 }
    }

    /// Which field this configures.
    #[must_use]
    pub fn id(&self) -> FieldId {
        self.id
    }

    /// The number of bits required to represent the modulus.
    #[must_use]
    pub fn field_bit_size(&self) -> u32 {
        debug_assert_eq!(self.id, FieldId::Bn254);
        FieldElement::MODULUS_BIT_SIZE
    }

    /// The field modulus, as canonical little-endian 64-bit limbs.
    #[must_use]
    pub fn modulus(&self) -> BigInt<4> {
        debug_assert_eq!(self.id, FieldId::Bn254);
        FieldElement::MODULUS
    }

    /// `2^exp` as a field element.
    #[must_use]
    pub fn two_pow(&self, exp: usize) -> FieldElement {
        debug_assert_eq!(self.id, FieldId::Bn254);
        FieldElement::two_pow(exp)
    }
}

impl Default for FieldConfig {
    fn default() -> FieldConfig {
        FieldConfig::bn254()
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use ark_ff::PrimeField;

    use super::*;

    #[test]
    fn bn254_config_reports_backing_constants() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.id(), FieldId::Bn254);
        assert_eq!(
            cfg.field_bit_size(),
            <ark_bn254::Fr as PrimeField>::MODULUS_BIT_SIZE
        );
        assert_eq!(cfg.modulus(), <ark_bn254::Fr as PrimeField>::MODULUS);
    }

    #[test]
    fn two_pow_matches_manual_powers() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.two_pow(0), FieldElement::ONE);
        assert_eq!(cfg.two_pow(1), FieldElement::from(2u64));
        for e in [2usize, 8, 31, 32, 63, 64, 200] {
            let mut expected = FieldElement::ONE;
            let two = FieldElement::from(2u64);
            for _ in 0..e {
                expected *= two;
            }
            assert_eq!(cfg.two_pow(e), expected);
        }
    }
}
