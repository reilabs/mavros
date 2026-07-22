//! The [`FieldElement`] abstraction over the concrete backing field.
//!
//! This is the compiler middle-end's field value type. It is a thin newtype over the arkworks bn254
//! scalar field that forwards every operation the middle-end needs 1:1, so that bn254 output stays
//! byte-identical while the rest of the compiler stops naming `ark_bn254::Fr` directly. It is
//! deliberately **not** `#[repr(transparent)]`.
//!
//! The witgen VM reinterprets raw frame memory as `Field` via pointer casts, and we keep that path
//! on the raw `ark_bn254::Fr` representation rather than pulling `FieldElement` into the VM.
//! Compiler code translates to/from the raw representation at the codegen boundary via
//! [`FieldElement::to_ark`] / `FieldElement::from` and the Montgomery-limb accessors.

use std::{
    fmt::{self, Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ff::{AdditiveGroup, BigInt, Field as _, PrimeField};

// TYPE ALIASES
// ================================================================================================

/// The backing field. All of `FieldElement`'s behavior is defined by forwarding to this type.
type Backing = ark_bn254::Fr;

// FIELD ELEMENT
// ================================================================================================

/// A field element in the compiler's middle-end.
///
/// Wraps the concrete backing field ([`ark_bn254::Fr`]) and forwards every operation to it. See the
/// module documentation for why this is not `#[repr(transparent)]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldElement(Backing);

// A derived `Debug` would print `FieldElement(..)`, which differs from the backing field's `Debug`
// and would perturb any `{:?}`-formatted field in a diffed dump or panic message. Forward instead.
impl Debug for FieldElement {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for FieldElement {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

// OPERATOR DEFINITION MACROS
// ================================================================================================

// Generates all four operand combinations (value/ref × value/ref) for a binary operator, each
// forwarding to the backing field.
//
// The backing field is `Copy`, so every combination reduces to the same by-value call.
macro_rules! forward_binop {
    ($trait:ident, $method:ident) => {
        impl $trait<FieldElement> for FieldElement {
            type Output = FieldElement;

            #[inline]
            fn $method(self, rhs: FieldElement) -> FieldElement {
                FieldElement($trait::$method(self.0, rhs.0))
            }
        }

        impl $trait<&FieldElement> for FieldElement {
            type Output = FieldElement;

            #[inline]
            fn $method(self, rhs: &FieldElement) -> FieldElement {
                FieldElement($trait::$method(self.0, rhs.0))
            }
        }

        impl $trait<FieldElement> for &FieldElement {
            type Output = FieldElement;

            #[inline]
            fn $method(self, rhs: FieldElement) -> FieldElement {
                FieldElement($trait::$method(self.0, rhs.0))
            }
        }

        impl $trait<&FieldElement> for &FieldElement {
            type Output = FieldElement;

            #[inline]
            fn $method(self, rhs: &FieldElement) -> FieldElement {
                FieldElement($trait::$method(self.0, rhs.0))
            }
        }
    };
}

macro_rules! forward_assign {
    ($trait:ident, $method:ident) => {
        impl $trait<FieldElement> for FieldElement {
            #[inline]
            fn $method(&mut self, rhs: FieldElement) {
                $trait::$method(&mut self.0, rhs.0);
            }
        }

        impl $trait<&FieldElement> for FieldElement {
            #[inline]
            fn $method(&mut self, rhs: &FieldElement) {
                $trait::$method(&mut self.0, rhs.0);
            }
        }
    };
}

// FIELD ELEMENT OPERATION DEFINITIONS
// ================================================================================================

forward_binop!(Add, add);
forward_binop!(Sub, sub);
forward_binop!(Mul, mul);
forward_binop!(Div, div);

forward_assign!(AddAssign, add_assign);
forward_assign!(SubAssign, sub_assign);
forward_assign!(MulAssign, mul_assign);

impl Neg for FieldElement {
    type Output = FieldElement;

    #[inline]
    fn neg(self) -> FieldElement {
        FieldElement(-self.0)
    }
}

impl Neg for &FieldElement {
    type Output = FieldElement;

    #[inline]
    fn neg(self) -> FieldElement {
        FieldElement(-self.0)
    }
}

impl Sum<FieldElement> for FieldElement {
    #[inline]
    fn sum<I: Iterator<Item = FieldElement>>(iter: I) -> FieldElement {
        FieldElement(iter.map(|x| x.0).sum())
    }
}

impl<'a> Sum<&'a FieldElement> for FieldElement {
    #[inline]
    fn sum<I: Iterator<Item = &'a FieldElement>>(iter: I) -> FieldElement {
        FieldElement(iter.map(|x| x.0).sum())
    }
}

// CONSTRUCTION AND CONVERSIONS
// ================================================================================================

// Forwards `From<int>` for every integer type the backing field supports.
//
// `i32` is particularly necessary as unsuffixed integer literals fall back to it.
macro_rules! forward_from_int {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<$ty> for FieldElement {
                #[inline]
                fn from(value: $ty) -> FieldElement {
                    FieldElement(Backing::from(value))
                }
            }
        )*
    };
}

forward_from_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, bool);

/// The boundary conversion from the raw backing field into a [`FieldElement`]. Used where the Noir
/// frontend hands a raw `ark_bn254::Fr` (via `into_repr()`) into a middle-end constant.
impl From<Backing> for FieldElement {
    #[inline]
    fn from(value: Backing) -> FieldElement {
        FieldElement(value)
    }
}

// `num_traits::Zero`/`One` are the traits arkworks re-exports as `ark_ff::Zero`/`One`, so a single
// impl of each satisfies call sites importing either path.
impl num_traits::Zero for FieldElement {
    #[inline]
    fn zero() -> FieldElement {
        FieldElement::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        num_traits::Zero::is_zero(&self.0)
    }
}

impl num_traits::One for FieldElement {
    #[inline]
    fn one() -> FieldElement {
        FieldElement::ONE
    }

    #[inline]
    fn is_one(&self) -> bool {
        num_traits::One::is_one(&self.0)
    }
}

// FIELD ELEMENT INHERENT API SURFACE
// ================================================================================================

impl FieldElement {
    /// The additive identity.
    pub const ZERO: FieldElement = FieldElement(<Backing as AdditiveGroup>::ZERO);

    /// The multiplicative identity.
    pub const ONE: FieldElement = FieldElement(<Backing as ark_ff::Field>::ONE);

    /// The field modulus, as canonical little-endian 64-bit limbs.
    pub const MODULUS: BigInt<4> = <Backing as PrimeField>::MODULUS;

    /// `(modulus - 1) / 2`, the threshold used for sign canonicalisation.
    pub const MODULUS_MINUS_ONE_DIV_TWO: BigInt<4> =
        <Backing as PrimeField>::MODULUS_MINUS_ONE_DIV_TWO;

    /// The number of bits required to represent the modulus.
    pub const MODULUS_BIT_SIZE: u32 = <Backing as PrimeField>::MODULUS_BIT_SIZE;

    /// Converts into the raw backing field.
    ///
    /// This is the middle-end → codegen boundary conversion; the returned value is the wrapped
    /// element verbatim, so downstream emission is byte-identical.
    #[inline]
    #[must_use]
    pub fn to_ark(self) -> Backing {
        self.0
    }

    /// The multiplicative inverse, or `None` for zero.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Option<FieldElement> {
        self.0.inverse().map(FieldElement)
    }

    /// Raises this element to `exp` (little-endian 64-bit limbs).
    #[inline]
    #[must_use]
    pub fn pow(&self, exp: impl AsRef<[u64]>) -> FieldElement {
        FieldElement(self.0.pow(exp))
    }

    /// `2^exp` as a field element (i.e. `2^exp mod p`). This is pure field arithmetic and so is
    /// correct for any field; it is the single home for the place-value scaling the lowering passes
    /// use for bit/byte decomposition and range reasoning.
    #[inline]
    #[must_use]
    pub fn two_pow(exp: usize) -> FieldElement {
        FieldElement::from(2u64).pow([exp as u64])
    }

    /// The canonical (non-Montgomery) integer representation, as an arkworks big integer.
    #[inline]
    #[must_use]
    pub fn into_bigint(&self) -> BigInt<4> {
        self.0.into_bigint()
    }

    /// Constructs from a canonical big integer, returning `None` if it is not less than the
    /// modulus.
    #[inline]
    #[must_use]
    pub fn from_bigint(bigint: BigInt<4>) -> Option<FieldElement> {
        Backing::from_bigint(bigint).map(FieldElement)
    }

    /// The raw Montgomery-form storage limbs.
    ///
    /// This is the representation the backends emit into bytecode words and the LLVM `[4 x i64]`
    /// field struct; it is **not** the canonical integer (use [`FieldElement::into_bigint`] for
    /// that).
    #[inline]
    #[must_use]
    pub fn montgomery_limbs(&self) -> [u64; 4] {
        self.0.0.0
    }

    /// Reconstructs a field element from its raw Montgomery-form storage limbs (no reduction).
    ///
    /// `new_unchecked` performs no reduction and assumes `limbs` is an already-canonical Montgomery
    /// encoding (`< p` in Montgomery form). That holds because the only inputs are values
    /// previously produced by [`FieldElement::montgomery_limbs`] or read back from the VM frame.
    #[inline]
    #[must_use]
    pub fn from_montgomery_limbs(limbs: [u64; 4]) -> FieldElement {
        FieldElement(Backing::new_unchecked(BigInt::new(limbs)))
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, Field, PrimeField, UniformRand};
    use num_traits::{One, Zero};

    use super::*;

    type Backing = ark_bn254::Fr;

    fn samples() -> Vec<Backing> {
        let mut rng = ark_std::test_rng();
        (0..256).map(|_| Backing::rand(&mut rng)).collect()
    }

    #[test]
    fn layout_is_not_forced_transparent_but_matches_backing_size() {
        // FieldElement carries exactly the backing field, so it is the same size / alignment even
        // though we deliberately do not rely on that via `repr(transparent)`.
        assert_eq!(size_of::<FieldElement>(), size_of::<Backing>());
        assert_eq!(align_of::<FieldElement>(), align_of::<Backing>());
    }

    #[test]
    fn to_ark_round_trips() {
        for f in samples() {
            assert_eq!(FieldElement::from(f).to_ark(), f);
            assert_eq!(
                FieldElement::from(FieldElement::from(f).to_ark()),
                FieldElement::from(f)
            );
        }
    }

    #[test]
    fn montgomery_limb_round_trip() {
        for f in samples() {
            let fe = FieldElement::from(f);
            assert_eq!(fe.montgomery_limbs(), f.0.0);
            assert_eq!(
                FieldElement::from_montgomery_limbs(fe.montgomery_limbs()),
                fe
            );
        }
    }

    #[test]
    fn canonical_bigint_round_trip() {
        for f in samples() {
            let fe = FieldElement::from(f);
            assert_eq!(fe.into_bigint(), f.into_bigint());
            assert_eq!(FieldElement::from_bigint(fe.into_bigint()), Some(fe));
        }
    }

    #[test]
    fn from_integers_matches_backing() {
        // Includes the unsuffixed-literal (`i32`) fallback path.
        assert_eq!(FieldElement::from(0).to_ark(), Backing::from(0));
        assert_eq!(FieldElement::from(1).to_ark(), Backing::from(1));
        assert_eq!(FieldElement::from(2).to_ark(), Backing::from(2));
        assert_eq!(FieldElement::from(255u8).to_ark(), Backing::from(255u8));
        assert_eq!(
            FieldElement::from(65535u16).to_ark(),
            Backing::from(65535u16)
        );
        assert_eq!(
            FieldElement::from(u32::MAX).to_ark(),
            Backing::from(u32::MAX)
        );
        assert_eq!(
            FieldElement::from(u64::MAX).to_ark(),
            Backing::from(u64::MAX)
        );
        assert_eq!(
            FieldElement::from(u128::MAX).to_ark(),
            Backing::from(u128::MAX)
        );
        assert_eq!(FieldElement::from(-1i64).to_ark(), Backing::from(-1i64));
        assert_eq!(FieldElement::from(-7i8).to_ark(), Backing::from(-7i8));
        assert_eq!(FieldElement::from(-7i16).to_ark(), Backing::from(-7i16));
        assert_eq!(FieldElement::from(-7i32).to_ark(), Backing::from(-7i32));
        assert_eq!(FieldElement::from(-7i128).to_ark(), Backing::from(-7i128));
        assert_eq!(FieldElement::from(true).to_ark(), Backing::from(true));
        assert_eq!(FieldElement::from(false).to_ark(), Backing::from(false));
    }

    #[test]
    fn arithmetic_matches_backing_all_operand_combos() {
        let mut rng = ark_std::test_rng();
        for _ in 0..1024 {
            let a = Backing::rand(&mut rng);
            let b = Backing::rand(&mut rng);
            let (fa, fb) = (FieldElement::from(a), FieldElement::from(b));

            // All four operand combinations of each operator must match the backing field.
            for (op_wrapped, op_raw) in [
                ((fa + fb, fa + &fb, &fa + fb, &fa + &fb), a + b),
                ((fa - fb, fa - &fb, &fa - fb, &fa - &fb), a - b),
                ((fa * fb, fa * &fb, &fa * fb, &fa * &fb), a * b),
            ] {
                let expected = FieldElement::from(op_raw);
                assert_eq!(op_wrapped.0, expected);
                assert_eq!(op_wrapped.1, expected);
                assert_eq!(op_wrapped.2, expected);
                assert_eq!(op_wrapped.3, expected);
            }

            if !b.is_zero() {
                let expected = FieldElement::from(a / b);
                assert_eq!(fa / fb, expected);
                assert_eq!(fa / &fb, expected);
                assert_eq!(&fa / fb, expected);
                assert_eq!(&fa / &fb, expected);
                // a / b == a * b^-1
                assert_eq!(fa / fb, fa * fb.inverse().unwrap());
            }

            assert_eq!((-fa), FieldElement::from(-a));
            assert_eq!((-&fa), FieldElement::from(-a));

            let mut acc = fa;
            acc += fb;
            assert_eq!(acc, FieldElement::from(a + b));
            let mut acc = fa;
            acc -= &fb;
            assert_eq!(acc, FieldElement::from(a - b));
            let mut acc = fa;
            acc *= fb;
            assert_eq!(acc, FieldElement::from(a * b));
        }
    }

    #[test]
    fn two_pow_matches_repeated_doubling() {
        assert_eq!(FieldElement::two_pow(0), FieldElement::ONE);
        for e in [1usize, 2, 8, 31, 32, 63, 64, 200, 253] {
            assert_eq!(
                FieldElement::two_pow(e),
                FieldElement::from(2u64).pow([e as u64])
            );
        }
    }

    #[test]
    fn constants_match_backing() {
        assert_eq!(FieldElement::ZERO, FieldElement::from(Backing::ZERO));
        assert_eq!(FieldElement::ONE, FieldElement::from(Backing::ONE));
        assert_eq!(<FieldElement as Zero>::zero(), FieldElement::ZERO);
        assert_eq!(<FieldElement as One>::one(), FieldElement::ONE);
        assert!(FieldElement::ZERO.is_zero());
        assert!(!FieldElement::ONE.is_zero());
        assert!(FieldElement::ONE.is_one());
        assert!(!FieldElement::ZERO.is_one());

        assert_eq!(FieldElement::MODULUS, <Backing as PrimeField>::MODULUS);
        assert_eq!(
            FieldElement::MODULUS_MINUS_ONE_DIV_TWO,
            <Backing as PrimeField>::MODULUS_MINUS_ONE_DIV_TWO
        );
        assert_eq!(
            FieldElement::MODULUS_BIT_SIZE,
            <Backing as PrimeField>::MODULUS_BIT_SIZE
        );

        // ZERO / ONE are usable in const context.
        const Z: FieldElement = FieldElement::ZERO;
        const O: FieldElement = FieldElement::ONE;
        assert_eq!(Z + O, FieldElement::ONE);
    }

    #[test]
    fn pow_and_inverse_match_backing() {
        for f in samples() {
            let fe = FieldElement::from(f);
            for e in [0u64, 1, 2, 5, 32, 253] {
                assert_eq!(fe.pow([e]), FieldElement::from(f.pow([e])));
            }
            assert_eq!(fe.inverse().map(FieldElement::to_ark), f.inverse());
        }
        assert_eq!(FieldElement::ZERO.inverse(), None);
    }

    #[test]
    fn hash_ord_eq_consistent_with_backing() {
        use std::collections::BTreeSet;

        let raw = samples();
        let wrapped: BTreeSet<FieldElement> = raw.iter().copied().map(FieldElement::from).collect();
        let raw_set: BTreeSet<Backing> = raw.iter().copied().collect();

        // Identical sorted iteration order — the property CSE interning depends on.
        let wrapped_sorted: Vec<Backing> = wrapped.iter().map(|f| f.to_ark()).collect();
        let raw_sorted: Vec<Backing> = raw_set.iter().copied().collect();
        assert_eq!(wrapped_sorted, raw_sorted);

        for a in &raw {
            for b in &raw {
                let (fa, fb) = (FieldElement::from(*a), FieldElement::from(*b));
                assert_eq!(fa == fb, a == b);
                assert_eq!(fa.cmp(&fb), a.cmp(b));
                assert_eq!(fa.partial_cmp(&fb), a.partial_cmp(b));
            }
        }
    }

    #[test]
    fn display_and_debug_are_transparent() {
        for f in samples() {
            let fe = FieldElement::from(f);
            assert_eq!(format!("{fe}"), format!("{f}"));
            assert_eq!(format!("{fe:?}"), format!("{f:?}"));
        }
    }

    #[test]
    fn sum_matches_backing() {
        let raw = samples();
        let expected: Backing = raw.iter().copied().sum();
        let wrapped: Vec<FieldElement> = raw.iter().copied().map(FieldElement::from).collect();
        assert_eq!(
            wrapped.iter().copied().sum::<FieldElement>(),
            FieldElement::from(expected)
        );
        assert_eq!(
            wrapped.iter().sum::<FieldElement>(),
            FieldElement::from(expected)
        );
    }
}
