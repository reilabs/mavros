//! The Goldilocks prime field, `p = 2^64 - 2^32 + 1`, stored as one Montgomery-form cell.
//!
//! The arithmetic is written out over `u64` and shares no generic code with bn254's arkworks
//! backing. A second field instantiated through arkworks' generics shares their limb-generic
//! helpers with bn254's, and the VM's `opt-level = "z"` build then outlines them from bn254's
//! handlers, costing bn254 programs 2–14% more instructions.
//!
//! The Montgomery radix is `R = 2^64`, so a storage cell is bit-for-bit arkworks' `Fp64` cell for
//! the same element, and the storage cell differs from the canonical integer as it does for bn254.
//! The reductions are in [`mavros_limb_arith::goldilocks`], which is shared with the wasm runtime.

use std::{
    cmp::Ordering,
    fmt,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use mavros_limb_arith::goldilocks::{self as arith, EPSILON};

use crate::field::{
    element::FieldElement,
    mavros_field::{FieldId, MavrosField, foreign_element},
};

// FIELD DEFINITION
// ================================================================================================

pub use mavros_limb_arith::goldilocks::MODULUS;

/// An element of the Goldilocks field, held as `x·R mod p` for the integer `x` it denotes.
///
/// Every operation keeps the cell below `p`. A value built from a frame cell past the modulus
/// ([`MavrosField::from_frame_cells`]) is kept as given and denotes no element.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Goldilocks(u64);

impl Goldilocks {
    /// The additive identity.
    pub const ZERO: Goldilocks = Goldilocks(0);

    /// The multiplicative identity.
    pub const ONE: Goldilocks = Goldilocks(EPSILON);

    /// The element denoting `value`, or `None` if it is not below the modulus.
    #[inline]
    #[must_use]
    pub const fn from_canonical(value: u64) -> Option<Goldilocks> {
        if value < MODULUS {
            Some(Goldilocks(arith::to_montgomery(value)))
        } else {
            None
        }
    }

    /// The canonical integer this element denotes.
    #[inline]
    #[must_use]
    pub const fn value(self) -> u64 {
        arith::from_montgomery(self.0)
    }

    /// The element's one storage cell, as a slice.
    #[inline]
    pub(crate) fn cells(&self) -> &[u64] {
        std::slice::from_ref(&self.0)
    }

    /// Whether this is the additive identity.
    #[inline]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Whether this is the multiplicative identity.
    #[inline]
    #[must_use]
    pub fn is_one(self) -> bool {
        self.0 == EPSILON
    }

    /// Raises this element to `exp` (little-endian 64-bit limbs).
    #[must_use]
    pub fn pow(self, exp: impl AsRef<[u64]>) -> Goldilocks {
        Goldilocks(arith::pow(self.0, exp.as_ref()))
    }

    /// The multiplicative inverse, or `None` for zero.
    #[must_use]
    pub fn inverse(self) -> Option<Goldilocks> {
        arith::inverse(self.0).map(Goldilocks)
    }
}

// ARITHMETIC
// ================================================================================================

impl Add for Goldilocks {
    type Output = Goldilocks;

    #[inline]
    fn add(self, rhs: Goldilocks) -> Goldilocks {
        Goldilocks(arith::add(self.0, rhs.0))
    }
}

impl Sub for Goldilocks {
    type Output = Goldilocks;

    #[inline]
    fn sub(self, rhs: Goldilocks) -> Goldilocks {
        Goldilocks(arith::sub(self.0, rhs.0))
    }
}

impl Mul for Goldilocks {
    type Output = Goldilocks;

    #[inline]
    fn mul(self, rhs: Goldilocks) -> Goldilocks {
        Goldilocks(arith::mul(self.0, rhs.0))
    }
}

impl Div for Goldilocks {
    type Output = Goldilocks;

    /// # Panics
    ///
    /// If `rhs` is zero.
    #[inline]
    fn div(self, rhs: Goldilocks) -> Goldilocks {
        self * rhs
            .inverse()
            .expect("division by zero in the Goldilocks field")
    }
}

impl Neg for Goldilocks {
    type Output = Goldilocks;

    #[inline]
    fn neg(self) -> Goldilocks {
        Goldilocks(arith::neg(self.0))
    }
}

impl AddAssign for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Goldilocks) {
        *self = *self + rhs;
    }
}

impl SubAssign for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Goldilocks) {
        *self = *self - rhs;
    }
}

impl MulAssign for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Goldilocks) {
        *self = *self * rhs;
    }
}

// CONVERSIONS AND FORMATTING
// ================================================================================================

impl From<u64> for Goldilocks {
    #[inline]
    fn from(value: u64) -> Goldilocks {
        Goldilocks(arith::to_montgomery(arith::reduce64(value)))
    }
}

impl From<u128> for Goldilocks {
    #[inline]
    fn from(value: u128) -> Goldilocks {
        Goldilocks(arith::to_montgomery(arith::reduce128(value)))
    }
}

impl From<bool> for Goldilocks {
    #[inline]
    fn from(value: bool) -> Goldilocks {
        Goldilocks::from(u64::from(value))
    }
}

macro_rules! from_unsigned {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for Goldilocks {
                #[inline]
                fn from(value: $ty) -> Goldilocks {
                    Goldilocks::from(u64::from(value))
                }
            }
        )*
    };
}

from_unsigned!(u8, u16, u32);

// A negative integer denotes `p - |value|`.
macro_rules! from_signed {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for Goldilocks {
                #[inline]
                fn from(value: $ty) -> Goldilocks {
                    let magnitude = Goldilocks::from(value.unsigned_abs());
                    if value < 0 { -magnitude } else { magnitude }
                }
            }
        )*
    };
}

from_signed!(i8, i16, i32, i64, i128);

// Ordered by the integers the elements denote.
impl Ord for Goldilocks {
    fn cmp(&self, other: &Goldilocks) -> Ordering {
        self.value().cmp(&other.value())
    }
}

impl PartialOrd for Goldilocks {
    fn partial_cmp(&self, other: &Goldilocks) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value(), f)
    }
}

impl fmt::Debug for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value(), f)
    }
}

// VM FIELD
// ================================================================================================

const _: () =
    assert!(size_of::<Goldilocks>() == 8 && align_of::<Goldilocks>() <= align_of::<u64>());

/// The Goldilocks field, stored as its one Montgomery-form cell.
///
/// # Safety
///
/// `repr(transparent)` over a `u64` makes the value exactly the cell `to_frame_cells` writes, every
/// pattern of it a valid `Goldilocks` (a plain integer, no niche), and `ZERO` the zero cell.
unsafe impl MavrosField for Goldilocks {
    const ID: FieldId = FieldId::Goldilocks;
    const STORAGE_CELLS: usize = 1;
    const CANONICAL_LIMBS: usize = 1;
    type Limbs = [u64; 1];
    const ZERO: Self = Goldilocks::ZERO;
    const ONE: Self = Goldilocks::ONE;

    fn inverse(&self) -> Option<Self> {
        Goldilocks::inverse(*self)
    }

    #[track_caller]
    fn from_element(value: FieldElement) -> Self {
        match value {
            FieldElement::Goldilocks(value) => value,
            _ => foreign_element(value, FieldId::Goldilocks),
        }
    }

    fn to_element(self) -> FieldElement {
        FieldElement::Goldilocks(self)
    }

    fn from_frame_cells(cells: &[u64]) -> Self {
        // Kept as given, so a punned `u64` or a cell past the modulus survives.
        Goldilocks(cells[0])
    }

    fn to_frame_cells(self, out: &mut [u64]) {
        out[0] = self.0;
    }

    fn storage_cell0(self) -> u64 {
        self.0
    }

    fn to_canonical_limbs(self) -> [u64; 1] {
        [self.value()]
    }

    fn from_canonical_limbs(limbs: &[u64]) -> Option<Self> {
        assert!(
            limbs.len() <= Self::CANONICAL_LIMBS,
            "{} limbs do not fit a Goldilocks element's {} canonical limb",
            limbs.len(),
            Self::CANONICAL_LIMBS
        );
        Goldilocks::from_canonical(limbs.first().copied().unwrap_or(0))
    }

    /// Rejection sampling: a draw of `p` or more is redrawn, which keeps the rest uniform.
    fn random<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        loop {
            if let Some(value) = Goldilocks::from_canonical(rng.next_u64()) {
                return value;
            }
        }
    }

    fn canonical_cmp(self, other: Self) -> Ordering {
        self.cmp(&other)
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use ark_ff::{BigInt, Field, Fp64, MontBackend, MontConfig, PrimeField};
    use ark_std::rand::RngCore;

    use super::*;

    #[derive(MontConfig)]
    #[modulus = "18446744069414584321"]
    #[generator = "7"]
    struct OracleConfig;

    /// arkworks' Goldilocks, the reference the hand-written arithmetic is checked against.
    type Oracle = Fp64<MontBackend<OracleConfig, 1>>;

    fn oracle(value: Goldilocks) -> Oracle {
        Oracle::from_bigint(BigInt([value.value()])).expect("a canonical value")
    }

    /// Whether `value` is `expected`, both as the integer and as the storage cell.
    fn agrees(value: Goldilocks, expected: Oracle) -> bool {
        value.value() == expected.into_bigint().0[0] && value.storage_cell0() == expected.0.0[0]
    }

    /// Values at every boundary the reductions branch on, and random ones.
    fn samples() -> Vec<Goldilocks> {
        let edges = [
            0,
            1,
            2,
            EPSILON - 1,
            EPSILON,
            EPSILON + 1,
            1 << 32,
            (1 << 63) - 1,
            1 << 63,
            (MODULUS - 1) / 2,
            (MODULUS + 1) / 2,
            MODULUS - 2,
            MODULUS - 1,
        ];
        let mut rng = ark_std::test_rng();
        edges
            .into_iter()
            .chain((0..400).map(|_| rng.next_u64() % MODULUS))
            .map(|value| Goldilocks::from_canonical(value).unwrap())
            .collect()
    }

    #[test]
    fn arithmetic_matches_arkworks() {
        let values = samples();
        for &a in &values {
            for &b in &values {
                assert!(agrees(a + b, oracle(a) + oracle(b)), "{a} + {b}");
                assert!(agrees(a - b, oracle(a) - oracle(b)), "{a} - {b}");
                assert!(agrees(a * b, oracle(a) * oracle(b)), "{a} * {b}");
                if !b.is_zero() {
                    assert!(agrees(a / b, oracle(a) / oracle(b)), "{a} / {b}");
                }
            }
            assert!(agrees(-a, -oracle(a)), "-{a}");
            match (a.inverse(), oracle(a).inverse()) {
                (Some(inverse), Some(expected)) => assert!(agrees(inverse, expected)),
                (None, None) => assert!(a.is_zero()),
                other => panic!("inverse of {a}: {other:?}"),
            }
            for exponent in [0, 1, 2, 7, 63, 64, MODULUS - 2, MODULUS - 1, u64::MAX] {
                assert!(agrees(a.pow([exponent]), oracle(a).pow([exponent])));
            }
            assert!(agrees(a.pow([3, 1]), oracle(a).pow([3, 1])));
        }
    }

    #[test]
    fn integers_reduce_as_arkworks_does() {
        let mut rng = ark_std::test_rng();
        // The last two borrow from the top half and then carry, a branch random inputs reach
        // about once in 2^32.
        let wide: Vec<u128> = [
            0,
            1,
            u128::from(MODULUS),
            u128::from(u64::MAX),
            u128::MAX,
            u128::MAX << 64,
            u128::MAX << 64 | 5,
        ]
        .into_iter()
        .chain((0..400).map(|_| u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())))
        .collect();
        for value in wide {
            assert!(
                agrees(Goldilocks::from(value), Oracle::from(value)),
                "{value}"
            );
            assert!(agrees(
                Goldilocks::from(value as u64),
                Oracle::from(value as u64)
            ));
            let signed = value as i128;
            assert!(
                agrees(Goldilocks::from(signed), Oracle::from(signed)),
                "{signed}"
            );
            assert!(agrees(
                Goldilocks::from(signed as i64),
                Oracle::from(signed as i64)
            ));
        }
        for value in [i8::MIN, -1, 0, 1, i8::MAX] {
            assert!(agrees(Goldilocks::from(value), Oracle::from(value)));
            assert!(agrees(
                Goldilocks::from(i32::from(value)),
                Oracle::from(i32::from(value))
            ));
        }
        assert!(agrees(Goldilocks::from(u8::MAX), Oracle::from(u8::MAX)));
        assert!(agrees(Goldilocks::from(u32::MAX), Oracle::from(u32::MAX)));
        assert_eq!(Goldilocks::from(true), Goldilocks::ONE);
        assert_eq!(Goldilocks::from(false), Goldilocks::ZERO);
    }

    #[test]
    fn goldilocks_storage_is_one_arkworks_montgomery_cell() {
        for value in samples() {
            let mut cells = [0u64; 1];
            value.to_frame_cells(&mut cells);
            assert_eq!(cells, oracle(value).0.0);
            // SAFETY: a `Goldilocks` is `repr(transparent)` over one `u64`.
            assert_eq!(unsafe { *(cells.as_ptr() as *const Goldilocks) }, value);
            assert_eq!(<Goldilocks as MavrosField>::from_frame_cells(&cells), value);
            assert_eq!(value.storage_cell0(), cells[0]);
            assert_eq!(value.to_canonical_limbs(), [value.value()]);
            assert_eq!(
                <Goldilocks as MavrosField>::from_canonical_limbs(&[value.value()]),
                Some(value)
            );
        }
        // Montgomery form is visible: `1` is not stored as the integer 1.
        assert_ne!(Goldilocks::ONE.storage_cell0(), 1);
        // A punned `u64`, and a cell past the modulus, are kept as given.
        for cell in [7, MODULUS, u64::MAX] {
            assert_eq!(
                <Goldilocks as MavrosField>::from_frame_cells(&[cell]).storage_cell0(),
                cell
            );
        }
        assert_eq!(
            <Goldilocks as MavrosField>::from_canonical_limbs(&[MODULUS]),
            None
        );
        assert_eq!(
            <Goldilocks as MavrosField>::from_canonical_limbs(&[]),
            Some(Goldilocks::ZERO)
        );
        assert_eq!(Goldilocks::from_canonical(MODULUS), None);
        assert_eq!(<Goldilocks as MavrosField>::ZERO.storage_cell0(), 0);
    }

    #[test]
    #[should_panic(expected = "do not fit")]
    fn goldilocks_from_canonical_limbs_rejects_too_many_limbs() {
        let _ = <Goldilocks as MavrosField>::from_canonical_limbs(&[0; 2]);
    }

    #[test]
    fn goldilocks_order_and_formatting_are_the_integers() {
        let values = samples();
        for &a in &values {
            for &b in values.iter().take(32) {
                assert_eq!(a.cmp(&b), a.value().cmp(&b.value()));
                assert_eq!(a.canonical_cmp(b), a.value().cmp(&b.value()));
            }
            assert_eq!(format!("{a}"), a.value().to_string());
            assert_eq!(format!("{a:?}"), a.value().to_string());
        }
    }

    #[test]
    fn goldilocks_elements_round_trip_through_field_element() {
        assert_eq!(<Goldilocks as MavrosField>::ID, FieldId::Goldilocks);
        for value in samples() {
            let element = value.to_element();
            assert_eq!(element, FieldElement::Goldilocks(value));
            assert_eq!(<Goldilocks as MavrosField>::from_element(element), value);
        }
    }

    #[test]
    #[should_panic(expected = "a Bn254 element where a Goldilocks one belongs")]
    fn goldilocks_from_element_refuses_a_bn254_element() {
        let _ = <Goldilocks as MavrosField>::from_element(FieldElement::Bn254(
            ark_bn254::Fr::from(1u64),
        ));
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn goldilocks_division_by_zero_panics() {
        let _ = Goldilocks::ONE / Goldilocks::ZERO;
    }

    #[test]
    fn goldilocks_field_constants() {
        assert_eq!(<Oracle as PrimeField>::MODULUS.0[0], MODULUS);
        assert_eq!(Goldilocks::from(u64::MAX).value(), EPSILON - 1);
        assert_eq!(Goldilocks::from(1u128 << 64).value(), EPSILON);
        assert_eq!(Goldilocks::from(1u128 << 96), -Goldilocks::ONE);
        assert_eq!(-Goldilocks::ONE + Goldilocks::ONE, Goldilocks::ZERO);
        assert_eq!(<Oracle as Field>::ONE, oracle(Goldilocks::ONE));
    }

    #[test]
    fn random_redraws_a_non_canonical_draw() {
        // `u64::MAX` is past the modulus, and the next draw wraps to `0`.
        let mut rng = rand::rngs::mock::StepRng::new(u64::MAX, 1);
        assert_eq!(
            <Goldilocks as MavrosField>::random(&mut rng),
            Goldilocks::ZERO
        );
    }

    #[test]
    fn try_from_canonical_limbs_refuses_without_panicking() {
        let from = <Goldilocks as MavrosField>::try_from_canonical_limbs;
        assert_eq!(from(&[MODULUS - 1]), Some(-Goldilocks::ONE));
        assert_eq!(from(&[MODULUS]), None);
        assert_eq!(from(&[0, 1]), None);
        assert_eq!(from(&[5, 0, 0]), Some(Goldilocks::from(5u64)));
    }
}
