//! Configuration for the field over which the program is running.
//!
//! [`FieldConfig`] carries the field's modulus, bit-width, and the derived constants the compiler
//! mints values from (e.g. `two_pow`). An instance is homed on the compiler's `SSA` and reached
//! through `ssa.field()` / `b.field()`, so the middle-end and R1CS generation mint and inspect
//! every field value through this object, and no pass names a concrete prime. The _values_ it
//! returns are bn254's (see the note on the `impl` below).

use ark_ff::BigInt;

use crate::field::{
    element::FieldElement,
    mavros_field::{FieldId, MavrosField},
};

// RUNTIME FIELD CONFIGURATION
// ================================================================================================

/// The configuration of the field a program operates over.
///
/// Currently bn254 is the only supported field, but this serves as a single configuration that is
/// threaded through the entire compiler so that operations get the necessary data from the
/// abstract interface, and none names a concrete field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldConfig {
    id: FieldId,
}

// The accessors below are **bn254-pinned in their values**: they are threaded and called
// everywhere, and every one of them answers with bn254's data.
//
// What guards that is the exhaustive `match self.id` each accessor is written as. It is deliberate
// that these matches carry a single arm and no `_` fallthrough: adding a variant to [`FieldId`] is
// then a _compile error_ at every accessor; a debug-build assertion would fire only if something
// happened to construct the new config. The type system is the forcing function.
//
// Two of them (`storage_cells`, `canonical_limbs`) already read their field's `MavrosField`
// constants, so a new field's arm names its type. Three (`field_bit_size`, `modulus`,
// `modulus_limbs`) return plain data and could be given real per-field arms as soon as that data
// exists. The rest return [`FieldElement`], which is a newtype over the bn254 backing field and so
// has no non-bn254 inhabitant to return; they cannot gain a truthful arm until P5 decides
// `FieldElement`'s multi-field shape.
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
        match self.id {
            FieldId::Bn254 => FieldElement::MODULUS_BIT_SIZE,
        }
    }

    /// The field modulus, as canonical little-endian 64-bit limbs.
    #[must_use]
    pub fn modulus(&self) -> BigInt<4> {
        match self.id {
            FieldId::Bn254 => FieldElement::MODULUS,
        }
    }

    /// The number of 64-bit cells one element occupies in a VM frame, a bytecode operand or the
    /// constant pool: the field's `MavrosField::STORAGE_CELLS`, for code that is not generic over
    /// the field.
    #[must_use]
    pub fn storage_cells(&self) -> usize {
        match self.id {
            FieldId::Bn254 => <ark_bn254::Fr as MavrosField>::STORAGE_CELLS,
        }
    }

    /// The number of 64-bit limbs of an element's canonical integer form: the field's
    /// `MavrosField::CANONICAL_LIMBS`, for code that is not generic over the field.
    #[must_use]
    pub fn canonical_limbs(&self) -> usize {
        match self.id {
            FieldId::Bn254 => <ark_bn254::Fr as MavrosField>::CANONICAL_LIMBS,
        }
    }

    /// `2^exp` as a field element.
    #[must_use]
    pub fn two_pow(&self, exp: usize) -> FieldElement {
        match self.id {
            FieldId::Bn254 => FieldElement::two_pow(exp),
        }
    }

    /// The field modulus as canonical little-endian 64-bit limbs (`modulus().0`).
    ///
    /// No `match` of its own: it delegates to [`FieldConfig::modulus`], which carries one.
    #[must_use]
    pub fn modulus_limbs(&self) -> [u64; 4] {
        self.modulus().0
    }

    /// The additive identity of the field.
    #[must_use]
    pub fn zero(&self) -> FieldElement {
        match self.id {
            FieldId::Bn254 => FieldElement::ZERO,
        }
    }

    /// The multiplicative identity of the field.
    #[must_use]
    pub fn one(&self) -> FieldElement {
        match self.id {
            FieldId::Bn254 => FieldElement::ONE,
        }
    }

    /// Embeds a value (an integer, `bool`, or existing element) into the field.
    #[must_use]
    pub fn constant(&self, value: impl Into<FieldElement>) -> FieldElement {
        match self.id {
            FieldId::Bn254 => value.into(),
        }
    }

    /// The field element with the given canonical integer representation, or `None` if the integer
    /// is not less than the modulus.
    #[must_use]
    pub fn from_bigint(&self, bigint: BigInt<4>) -> Option<FieldElement> {
        match self.id {
            FieldId::Bn254 => FieldElement::from_bigint(bigint),
        }
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
        assert_eq!(cfg.storage_cells(), 4);
        assert_eq!(cfg.canonical_limbs(), 4);
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

    #[test]
    fn instance_accessors_match_backing_statics() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.zero(), FieldElement::ZERO);
        assert_eq!(cfg.one(), FieldElement::ONE);
        assert_eq!(cfg.modulus_limbs(), FieldElement::MODULUS.0);
        assert_eq!(cfg.modulus_limbs(), cfg.modulus().0);
    }

    #[test]
    fn constant_matches_from() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.constant(0u64), FieldElement::ZERO);
        assert_eq!(cfg.constant(1u64), FieldElement::ONE);
        for n in [2u64, 7, 255, 1 << 40] {
            assert_eq!(cfg.constant(n), FieldElement::from(n));
        }
        assert_eq!(cfg.constant(-1i64), FieldElement::from(-1i64));
        // An already-constructed element passes through unchanged.
        assert_eq!(
            cfg.constant(FieldElement::from(42u64)),
            FieldElement::from(42u64)
        );
    }

    #[test]
    fn from_bigint_matches_backing() {
        let cfg = FieldConfig::bn254();
        let seven = FieldElement::from(7u64).into_bigint();
        assert_eq!(cfg.from_bigint(seven), FieldElement::from_bigint(seven));
        assert_eq!(cfg.from_bigint(seven), Some(FieldElement::from(7u64)));
        // At or above the modulus is rejected.
        assert_eq!(cfg.from_bigint(cfg.modulus()), None);
    }
}
