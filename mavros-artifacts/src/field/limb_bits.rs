//! The width of the limbs a field's wide integers are cut into.

use mavros_int_semantics::int_bits::HOST_LIMB_BITS;
use serde::{Deserialize, Serialize};

use crate::field::config::FieldConfig;

/// The narrowest limb a wide witnessed integer can be cut at.
/// Decomposition halves a limb, so a narrower one would have no exact half.
pub const NARROWEST_LIMB_BITS: usize = 2;

/// The width of the limbs an integer too wide for one element of a field is cut into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LimbBits(pub(crate) usize);

impl LimbBits {
    /// `field`'s widest limb, which a program is cut at unless the compiler narrows it.
    #[must_use]
    pub fn widest(field: FieldConfig) -> Self {
        LimbBits(
            widest_for_modulus_bits(field.field_bit_size() as usize)
                .unwrap_or_else(|| panic!("{:?} has no witness limb", field.id())),
        )
    }

    /// `bits`, if `field` can be cut at it.
    pub fn new(field: FieldConfig, bits: usize) -> Result<Self, String> {
        let mut allowed = Self::candidates(field).map(LimbBits::bits);
        if allowed.any(|allowed| allowed == bits) {
            Ok(LimbBits(bits))
        } else {
            Err(format!(
                "{bits} bits is not a witness limb {:?} can be cut at; the allowed widths are \
                 {:?}",
                field.id(),
                Self::candidates(field)
                    .map(LimbBits::bits)
                    .collect::<Vec<_>>()
            ))
        }
    }

    /// The limbs `field` can be cut at, widest to narrowest.
    pub fn candidates(field: FieldConfig) -> impl Iterator<Item = Self> {
        candidates_for_widest(Self::widest(field).0).map(LimbBits)
    }

    #[must_use]
    pub fn bits(self) -> usize {
        self.0
    }
}

impl From<LimbBits> for usize {
    fn from(limb_bits: LimbBits) -> Self {
        limb_bits.0
    }
}

/// The widest limb of a field whose modulus has `modulus_bits` bits: the widest power of two `h`
/// up to [`HOST_LIMB_BITS`] with `2h` no wider than `modulus_bits - 1`, the widest integer an
/// element holds injectively, so that every product of two limbs is a distinct element. [`None`]
/// below a five-bit modulus, where no limb of [`NARROWEST_LIMB_BITS`] is injective.
#[must_use]
pub fn widest_for_modulus_bits(modulus_bits: usize) -> Option<usize> {
    let injective = modulus_bits.saturating_sub(1);
    candidates_for_widest(HOST_LIMB_BITS).find(|bits| 2 * bits <= injective)
}

/// The limbs a field whose widest limb is `widest` can be cut at, widest to narrowest: `widest`
/// and each of its halvings down to [`NARROWEST_LIMB_BITS`].
pub fn candidates_for_widest(widest: usize) -> impl Iterator<Item = usize> {
    std::iter::successors(Some(widest), |bits| {
        (bits / 2 >= NARROWEST_LIMB_BITS).then_some(bits / 2)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_is_cut_at_its_widest_limb_and_its_halvings() {
        let limbs = |field| {
            LimbBits::candidates(field)
                .map(LimbBits::bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(limbs(FieldConfig::bn254()), [64, 32, 16, 8, 4, 2]);
        assert_eq!(limbs(FieldConfig::goldilocks()), [16, 8, 4, 2]);
    }

    #[test]
    fn a_limb_outside_the_candidates_is_refused() {
        let bn254 = FieldConfig::bn254();
        for bits in [0, 1, 3, 48, 128] {
            let error = LimbBits::new(bn254, bits).unwrap_err();
            assert!(error.contains("not a witness limb"), "{error}");
        }
        assert!(LimbBits::new(FieldConfig::goldilocks(), 32).is_err());
        assert_eq!(LimbBits::new(bn254, 16).unwrap().bits(), 16);
    }

    #[test]
    fn the_widest_limb_squares_into_the_injective_width() {
        assert_eq!(widest_for_modulus_bits(254), Some(64));
        assert_eq!(widest_for_modulus_bits(64), Some(16));
        assert_eq!(widest_for_modulus_bits(65), Some(32));
        assert_eq!(widest_for_modulus_bits(5), Some(2));
        assert_eq!(widest_for_modulus_bits(4), None);
    }
}
