//! Configuration for the field over which the program is running.
//!
//! [`FieldConfig`] carries the field's modulus, bit-width, and the derived constants the compiler
//! mints values from (e.g. `two_pow`). An instance is homed on the compiler's `SSA` and reached
//! through `ssa.field()` / `b.field()`, so the middle-end mints and inspects every field value
//! through this object rather than through static associated items — no pass names a concrete
//! prime. The *values* it returns are still bn254's for the moment (see the note on the `impl`
//! below); codegen and the VM also currently remain on the raw representation and do not consult
//! it.

use ark_ff::BigInt;

use crate::field::{element::FieldElement, mavros_field::FieldId};

// RUNTIME FIELD CONFIGURATION
// ================================================================================================

/// The configuration of the field a program operates over.
///
/// bn254 is the only field today, so this is effectively a tag; its value is that it is a single
/// object threaded through the SSA and the symbolic evaluators, which a second field can later be
/// taught to answer differently without touching any of those call sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldConfig {
    id: FieldId,
}

// The constant accessors below are **bn254-pinned in their values**: they are threaded and called
// everywhere, but every one of them still answers with bn254's constants.
//
// What guards that is the exhaustive `match self.id` each accessor is written as. It is deliberate
// that these matches carry a single arm and no `_` fallthrough: adding a variant to [`FieldId`] is
// then a *compile error* at every accessor, rather than a debug-build-only assertion that fires
// only if something happens to construct the new config. The type system is the forcing function.
//
// Three of them (`field_bit_size`, `modulus`, `modulus_limbs`) return plain data and could be given
// real per-field arms as soon as that data exists. The rest return [`FieldElement`], which is a
// newtype over the bn254 backing field and so has no non-bn254 inhabitant to return; they cannot
// gain a truthful arm until P5 decides `FieldElement`'s multi-field shape.
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
