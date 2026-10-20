//! Configuration for the field over which the program is running.
//!
//! [`FieldConfig`] carries the field's modulus, bit-width, and the derived constants the compiler
//! mints values from (e.g. `two_pow`). An instance is homed on the compiler's `SSA` and reached
//! through `ssa.field()` / `b.field()`, so the middle-end and R1CS generation mint and inspect
//! every field value through this object, and no pass names a concrete prime.

use ark_ff::{AdditiveGroup, BigInt, PrimeField};

use crate::field::{
    element::FieldElement,
    goldilocks::{self, Goldilocks},
    mavros_field::{FieldId, MavrosField},
};

// RUNTIME FIELD CONFIGURATION
// ================================================================================================

/// The configuration of the field a program operates over.
///
/// It is threaded through the entire compiler so that operations get the necessary data from the
/// abstract interface, and none names a concrete field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldConfig {
    id: FieldId,
}

// Every accessor that depends on the field, and [`IntoFieldElement`]'s integer embedding, is an
// exhaustive `match self.id` with no `_` fallthrough or delegates to one. Adding a variant to
// [`FieldId`] is then a _compile error_ at every such site; a debug-build assertion would fire only
// if something happened to construct the new config. The type system is the forcing function.
impl FieldConfig {
    /// The bn254 configuration.
    #[must_use]
    pub const fn bn254() -> FieldConfig {
        FieldConfig { id: FieldId::Bn254 }
    }

    /// The Goldilocks configuration.
    #[must_use]
    pub const fn goldilocks() -> FieldConfig {
        FieldConfig {
            id: FieldId::Goldilocks,
        }
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
            FieldId::Bn254 => <ark_bn254::Fr as PrimeField>::MODULUS_BIT_SIZE,
            FieldId::Goldilocks => u64::BITS - goldilocks::MODULUS.leading_zeros(),
        }
    }

    /// The field modulus, as canonical little-endian 64-bit limbs.
    #[must_use]
    pub fn modulus(&self) -> BigInt<4> {
        match self.id {
            FieldId::Bn254 => <ark_bn254::Fr as PrimeField>::MODULUS,
            FieldId::Goldilocks => BigInt([goldilocks::MODULUS, 0, 0, 0]),
        }
    }

    /// The number of 64-bit cells one element occupies in a VM frame, a bytecode operand or the
    /// constant pool: the field's `MavrosField::STORAGE_CELLS`, for code that is not generic over
    /// the field.
    #[must_use]
    pub fn storage_cells(&self) -> usize {
        match self.id {
            FieldId::Bn254 => <ark_bn254::Fr as MavrosField>::STORAGE_CELLS,
            FieldId::Goldilocks => <Goldilocks as MavrosField>::STORAGE_CELLS,
        }
    }

    /// The number of 64-bit limbs of an element's canonical integer form: the field's
    /// `MavrosField::CANONICAL_LIMBS`, for code that is not generic over the field.
    #[must_use]
    pub fn canonical_limbs(&self) -> usize {
        match self.id {
            FieldId::Bn254 => <ark_bn254::Fr as MavrosField>::CANONICAL_LIMBS,
            FieldId::Goldilocks => <Goldilocks as MavrosField>::CANONICAL_LIMBS,
        }
    }

    /// `2^exp` as a field element.
    ///
    /// Delegates to [`FieldConfig::constant`], whose integer embedding holds the exhaustive
    /// `match`.
    #[must_use]
    pub fn two_pow(&self, exp: usize) -> FieldElement {
        self.constant(2u64).pow([exp as u64])
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
            FieldId::Bn254 => FieldElement::Bn254(<ark_bn254::Fr as AdditiveGroup>::ZERO),
            FieldId::Goldilocks => FieldElement::Goldilocks(Goldilocks::ZERO),
        }
    }

    /// The multiplicative identity of the field.
    #[must_use]
    pub fn one(&self) -> FieldElement {
        match self.id {
            FieldId::Bn254 => FieldElement::Bn254(<ark_bn254::Fr as ark_ff::Field>::ONE),
            FieldId::Goldilocks => FieldElement::Goldilocks(Goldilocks::ONE),
        }
    }

    /// Embeds a value into the field: an integer or `bool` reduces modulo the modulus, and an
    /// element (a [`FieldElement`] or a raw `ark_bn254::Fr`/`Goldilocks`) passes through.
    ///
    /// # Panics
    ///
    /// If `value` is an element of another field.
    #[must_use]
    #[track_caller]
    pub fn constant(&self, value: impl IntoFieldElement) -> FieldElement {
        value.into_field_element(*self)
    }

    /// The field element with the given canonical integer representation, or `None` if the integer
    /// is not less than the modulus.
    #[must_use]
    pub fn from_bigint(&self, bigint: BigInt<4>) -> Option<FieldElement> {
        match self.id {
            FieldId::Bn254 => ark_bn254::Fr::from_bigint(bigint).map(FieldElement::Bn254),
            // A canonical Goldilocks integer has one limb, so a non-zero upper limb is past the
            // modulus.
            FieldId::Goldilocks => match bigint.0 {
                [limb, 0, 0, 0] => Goldilocks::from_canonical(limb).map(FieldElement::Goldilocks),
                _ => None,
            },
        }
    }
}

// EMBEDDING VALUES
// ================================================================================================

/// A value [`FieldConfig::constant`] can embed into a field.
pub trait IntoFieldElement {
    /// This value as an element of `field`.
    fn into_field_element(self, field: FieldConfig) -> FieldElement;
}

// An integer reduces modulo the field's modulus, so a negative integer denotes `p - |value|`.
//
// `i32` is particularly necessary as unsuffixed integer literals fall back to it.
macro_rules! embed_int {
    ($($ty:ty),* $(,)?) => {
        $(
            impl IntoFieldElement for $ty {
                #[inline]
                fn into_field_element(self, field: FieldConfig) -> FieldElement {
                    match field.id {
                        FieldId::Bn254 => FieldElement::Bn254(ark_bn254::Fr::from(self)),
                        FieldId::Goldilocks => FieldElement::Goldilocks(Goldilocks::from(self)),
                    }
                }
            }
        )*
    };
}

embed_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, bool);

// A raw element converts to its field's `FieldElement`, so it embeds only into that field.
macro_rules! embed_raw {
    ($($ty:ty),* $(,)?) => {
        $(
            impl IntoFieldElement for $ty {
                #[inline]
                #[track_caller]
                fn into_field_element(self, field: FieldConfig) -> FieldElement {
                    FieldElement::from(self).into_field_element(field)
                }
            }
        )*
    };
}

embed_raw!(ark_bn254::Fr, Goldilocks);

/// An element passes through unchanged when it belongs to the field.
impl IntoFieldElement for FieldElement {
    #[inline]
    #[track_caller]
    fn into_field_element(self, field: FieldConfig) -> FieldElement {
        assert!(
            self.field_id() == field.id(),
            "ICE: embedding {self:?} into {:?}, which it does not belong to",
            field.id
        );
        self
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    type Fr = ark_bn254::Fr;

    #[test]
    fn bn254_config_reports_backing_constants() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.id(), FieldId::Bn254);
        assert_eq!(cfg.field_bit_size(), <Fr as PrimeField>::MODULUS_BIT_SIZE);
        assert_eq!(cfg.modulus(), <Fr as PrimeField>::MODULUS);
        assert_eq!(cfg.modulus_limbs(), cfg.modulus().0);
        assert_eq!(cfg.storage_cells(), 4);
        assert_eq!(cfg.canonical_limbs(), 4);
    }

    #[test]
    fn goldilocks_config_reports_backing_constants() {
        let cfg = FieldConfig::goldilocks();
        assert_eq!(cfg.id(), FieldId::Goldilocks);
        assert_eq!(cfg.field_bit_size(), 64);
        assert_eq!(cfg.modulus(), BigInt([goldilocks::MODULUS, 0, 0, 0]));
        assert_eq!(cfg.storage_cells(), 1);
        assert_eq!(cfg.canonical_limbs(), 1);
    }

    #[test]
    fn goldilocks_constants_reduce_modulo_p() {
        let cfg = FieldConfig::goldilocks();
        let gl = |value: u64| FieldElement::from(Goldilocks::from(value));
        assert_eq!(cfg.zero(), gl(0));
        assert_eq!(cfg.one(), gl(1));
        assert_eq!(cfg.constant(7u8), gl(7));
        assert_eq!(cfg.constant(-1i64), gl(goldilocks::MODULUS - 1));
        assert_eq!(cfg.constant(goldilocks::MODULUS), cfg.zero());
        // `2^64 = 2^32 - 1 (mod p)`.
        assert_eq!(cfg.constant(u64::MAX), gl((1 << 32) - 2));
        assert_eq!(cfg.two_pow(64), gl((1 << 32) - 1));
        assert_eq!(cfg.constant(true), cfg.one());
        let element = cfg.constant(42u64);
        assert_eq!(cfg.constant(element), element);
    }

    #[test]
    #[should_panic(expected = "which it does not belong to")]
    fn goldilocks_constant_refuses_a_bn254_element() {
        let _ = FieldConfig::goldilocks().constant(Fr::from(7u64));
    }

    #[test]
    fn goldilocks_from_bigint_takes_one_limb_below_the_modulus() {
        let cfg = FieldConfig::goldilocks();
        let top = BigInt([goldilocks::MODULUS - 1, 0, 0, 0]);
        assert_eq!(cfg.from_bigint(top), Some(-cfg.one()));
        assert_eq!(cfg.from_bigint(cfg.modulus()), None);
        // A non-zero upper limb is past the modulus, whatever the low limb holds.
        for upper in 1..4 {
            let mut limbs = [7, 0, 0, 0];
            limbs[upper] = 1;
            assert_eq!(cfg.from_bigint(BigInt(limbs)), None);
        }
        let seven = cfg.constant(7u64);
        assert_eq!(cfg.from_bigint(seven.into_bigint()), Some(seven));
    }

    #[test]
    fn two_pow_matches_manual_powers() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.two_pow(0), cfg.one());
        assert_eq!(cfg.two_pow(1), FieldElement::from(Fr::from(2u64)));
        for e in [2usize, 8, 31, 32, 63, 64, 200, 253, 254] {
            let mut expected = Fr::from(1u64);
            for _ in 0..e {
                expected *= Fr::from(2u64);
            }
            assert_eq!(cfg.two_pow(e), FieldElement::from(expected));
        }
    }

    #[test]
    fn identities_are_the_backing_identities() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.zero(), FieldElement::from(<Fr as AdditiveGroup>::ZERO));
        assert_eq!(cfg.one(), FieldElement::from(<Fr as ark_ff::Field>::ONE));
    }

    #[test]
    fn constant_matches_backing_from() {
        let cfg = FieldConfig::bn254();
        assert_eq!(cfg.constant(0u64), cfg.zero());
        assert_eq!(cfg.constant(1u64), cfg.one());
        // Includes the unsuffixed-literal (`i32`) fallback path.
        assert_eq!(cfg.constant(2), FieldElement::from(Fr::from(2)));
        for n in [7u64, 255, 1 << 40, u64::MAX] {
            assert_eq!(cfg.constant(n), FieldElement::from(Fr::from(n)));
        }
        assert_eq!(cfg.constant(255u8), FieldElement::from(Fr::from(255u8)));
        assert_eq!(
            cfg.constant(65535u16),
            FieldElement::from(Fr::from(65535u16))
        );
        assert_eq!(
            cfg.constant(u32::MAX),
            FieldElement::from(Fr::from(u32::MAX))
        );
        assert_eq!(
            cfg.constant(u128::MAX),
            FieldElement::from(Fr::from(u128::MAX))
        );
        assert_eq!(cfg.constant(-1i64), -cfg.one());
        assert_eq!(cfg.constant(-7i8), FieldElement::from(Fr::from(-7i8)));
        assert_eq!(cfg.constant(-7i16), FieldElement::from(Fr::from(-7i16)));
        assert_eq!(cfg.constant(-7i32), FieldElement::from(Fr::from(-7i32)));
        assert_eq!(cfg.constant(-7i128), FieldElement::from(Fr::from(-7i128)));
        assert_eq!(cfg.constant(true), cfg.one());
        assert_eq!(cfg.constant(false), cfg.zero());
        // An element of the field passes through unchanged.
        let forty_two = cfg.constant(42u64);
        assert_eq!(cfg.constant(forty_two), forty_two);
    }

    #[test]
    fn constant_passes_a_raw_element_of_the_field_through() {
        let cfg = FieldConfig::bn254();
        assert_eq!(
            cfg.constant(Fr::from(7u64)),
            FieldElement::from(Fr::from(7u64))
        );
    }

    #[test]
    #[should_panic(expected = "which it does not belong to")]
    fn constant_refuses_a_raw_element_of_another_field() {
        let _ = FieldConfig::bn254().constant(Goldilocks::from(7u64));
    }

    #[test]
    #[should_panic(expected = "which it does not belong to")]
    fn constant_refuses_an_element_of_another_field() {
        let _ = FieldConfig::bn254().constant(FieldElement::from(Goldilocks::from(7u64)));
    }

    #[test]
    fn from_bigint_matches_backing() {
        let cfg = FieldConfig::bn254();
        let seven = cfg.constant(7u64).into_bigint();
        assert_eq!(cfg.from_bigint(seven), Some(cfg.constant(7u64)));
        // At or above the modulus is rejected.
        assert_eq!(cfg.from_bigint(cfg.modulus()), None);
        let mut below = cfg.modulus();
        below.0[0] -= 1;
        assert_eq!(cfg.from_bigint(below), Some(-cfg.one()));
        // Every element's canonical integer round-trips.
        let mut rng = ark_std::test_rng();
        for _ in 0..256 {
            let value = FieldElement::from(<Fr as ark_ff::UniformRand>::rand(&mut rng));
            assert_eq!(cfg.from_bigint(value.into_bigint()), Some(value));
        }
    }
}
