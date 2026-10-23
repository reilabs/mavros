//! The [`FieldElement`] abstraction over the supported fields.
//!
//! This is the compiler middle-end's field value type: one variant per supported field, each
//! wrapping that field's raw type (bn254's arkworks `Fr`, the hand-written [`Goldilocks`]) and
//! forwarding every operation the middle-end needs 1:1, so bn254 output is byte-identical to
//! computing on `Fr` directly and the rest of the compiler names no concrete field. The field is
//! chosen once per compilation (by the program's `FieldConfig`), so every element of one program
//! carries the same variant, and an operation on two different variants is a compiler bug that
//! panics.
//!
//! The middle-end mints its constants through `FieldConfig` (`zero`, `one`, `constant`, `two_pow`,
//! `from_bigint`), which knows the field. `FieldElement` itself has no field-less constructor
//! besides the `From` conversions out of each field's raw type.
//!
//! The witgen VM reinterprets raw frame memory as the raw field type via pointer casts, so it
//! computes on the raw type and never sees a `FieldElement`. Compiler code translates at the
//! boundary with `MavrosField::from_element` / `MavrosField::to_element` and
//! [`FieldElement::storage_cells`].

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ff::{BigInt, Field as _, PrimeField};

use crate::field::{goldilocks::Goldilocks, mavros_field::FieldId};

// FIELD ELEMENT
// ================================================================================================

/// A field element in the compiler's middle-end, tagged with its field.
#[derive(Clone, Copy, Eq)]
pub enum FieldElement {
    /// An element of the bn254 scalar field.
    Bn254(ark_bn254::Fr),

    /// An element of the Goldilocks field.
    Goldilocks(Goldilocks),
}

/// Panics for an operation on elements of two different fields.
#[cold]
#[track_caller]
fn mixed_fields(lhs: FieldElement, rhs: FieldElement) -> ! {
    panic!(
        "ICE: an operation on field elements of two fields ({:?} and {:?})",
        lhs.field_id(),
        rhs.field_id()
    )
}

// `Debug` forwards to the backing field: a derived one would print `Bn254(..)` and perturb every
// `{:?}`-formatted field in a diffed dump or panic message.
impl Debug for FieldElement {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldElement::Bn254(value) => Debug::fmt(value, f),
            FieldElement::Goldilocks(value) => Debug::fmt(value, f),
        }
    }
}

impl Display for FieldElement {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldElement::Bn254(value) => Display::fmt(value, f),
            FieldElement::Goldilocks(value) => Display::fmt(value, f),
        }
    }
}

// Hashes the value alone, so a bn254 element hashes exactly as its raw `Fr` does and hash-ordered
// output matches the raw field's. Only elements of one field are ever equal, so omitting the tag
// keeps `Hash` consistent with `Eq`.
impl Hash for FieldElement {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            FieldElement::Bn254(value) => value.hash(state),
            FieldElement::Goldilocks(value) => value.hash(state),
        }
    }
}

// Compares by the backing field's equality. Two fields' elements meeting is a compiler bug, so
// comparing them panics like any other mixed-field operation; answering `false` would let a
// mixed-field constant check silently fail to fold.
impl PartialEq for FieldElement {
    #[inline]
    #[track_caller]
    fn eq(&self, other: &FieldElement) -> bool {
        match (self, other) {
            (FieldElement::Bn254(a), FieldElement::Bn254(b)) => a == b,
            (FieldElement::Goldilocks(a), FieldElement::Goldilocks(b)) => a == b,
            _ => mixed_fields(*self, *other),
        }
    }
}

// Orders by the backing field's order. Elements of two fields have no order, so comparing them
// panics like any other mixed-field operation.
impl Ord for FieldElement {
    #[inline]
    #[track_caller]
    fn cmp(&self, other: &FieldElement) -> Ordering {
        match (self, other) {
            (FieldElement::Bn254(a), FieldElement::Bn254(b)) => a.cmp(b),
            (FieldElement::Goldilocks(a), FieldElement::Goldilocks(b)) => a.cmp(b),
            _ => mixed_fields(*self, *other),
        }
    }
}

impl PartialOrd for FieldElement {
    #[inline]
    fn partial_cmp(&self, other: &FieldElement) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// OPERATOR DEFINITION MACROS
// ================================================================================================

// Generates all four operand combinations (value/ref × value/ref) for a binary operator, each
// forwarding to the backing field of matching operands.
//
// Every backing field is `Copy`, so the reference combinations reduce to the by-value one.
macro_rules! forward_binop {
    ($trait:ident, $method:ident) => {
        impl $trait<FieldElement> for FieldElement {
            type Output = FieldElement;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: FieldElement) -> FieldElement {
                match (self, rhs) {
                    (FieldElement::Bn254(a), FieldElement::Bn254(b)) => {
                        FieldElement::Bn254($trait::$method(a, b))
                    }
                    (FieldElement::Goldilocks(a), FieldElement::Goldilocks(b)) => {
                        FieldElement::Goldilocks($trait::$method(a, b))
                    }
                    _ => mixed_fields(self, rhs),
                }
            }
        }

        impl $trait<&FieldElement> for FieldElement {
            type Output = FieldElement;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: &FieldElement) -> FieldElement {
                $trait::$method(self, *rhs)
            }
        }

        impl $trait<FieldElement> for &FieldElement {
            type Output = FieldElement;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: FieldElement) -> FieldElement {
                $trait::$method(*self, rhs)
            }
        }

        impl $trait<&FieldElement> for &FieldElement {
            type Output = FieldElement;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: &FieldElement) -> FieldElement {
                $trait::$method(*self, *rhs)
            }
        }
    };
}

macro_rules! forward_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<FieldElement> for FieldElement {
            #[inline]
            #[track_caller]
            fn $method(&mut self, rhs: FieldElement) {
                *self = *self $op rhs;
            }
        }

        impl $trait<&FieldElement> for FieldElement {
            #[inline]
            #[track_caller]
            fn $method(&mut self, rhs: &FieldElement) {
                *self = *self $op *rhs;
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

forward_assign!(AddAssign, add_assign, +);
forward_assign!(SubAssign, sub_assign, -);
forward_assign!(MulAssign, mul_assign, *);

impl Neg for FieldElement {
    type Output = FieldElement;

    #[inline]
    fn neg(self) -> FieldElement {
        match self {
            FieldElement::Bn254(value) => FieldElement::Bn254(-value),
            FieldElement::Goldilocks(value) => FieldElement::Goldilocks(-value),
        }
    }
}

impl Neg for &FieldElement {
    type Output = FieldElement;

    #[inline]
    fn neg(self) -> FieldElement {
        -*self
    }
}

// CONSTRUCTION AND CONVERSIONS
// ================================================================================================

/// A raw bn254 element as a [`FieldElement`] of bn254.
impl From<ark_bn254::Fr> for FieldElement {
    #[inline]
    fn from(value: ark_bn254::Fr) -> FieldElement {
        FieldElement::Bn254(value)
    }
}

impl From<Goldilocks> for FieldElement {
    #[inline]
    fn from(value: Goldilocks) -> FieldElement {
        FieldElement::Goldilocks(value)
    }
}

// FIELD ELEMENT INHERENT API SURFACE
// ================================================================================================

impl FieldElement {
    /// The field this element belongs to.
    #[inline]
    #[must_use]
    pub fn field_id(&self) -> FieldId {
        match self {
            FieldElement::Bn254(_) => FieldId::Bn254,
            FieldElement::Goldilocks(_) => FieldId::Goldilocks,
        }
    }

    /// Whether this is the additive identity.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match self {
            FieldElement::Bn254(value) => num_traits::Zero::is_zero(value),
            FieldElement::Goldilocks(value) => value.is_zero(),
        }
    }

    /// Whether this is the multiplicative identity.
    #[inline]
    #[must_use]
    pub fn is_one(&self) -> bool {
        match self {
            FieldElement::Bn254(value) => num_traits::One::is_one(value),
            FieldElement::Goldilocks(value) => value.is_one(),
        }
    }

    /// The multiplicative inverse, or `None` for zero.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Option<FieldElement> {
        match self {
            FieldElement::Bn254(value) => value.inverse().map(FieldElement::Bn254),
            FieldElement::Goldilocks(value) => value.inverse().map(FieldElement::Goldilocks),
        }
    }

    /// Raises this element to `exp` (little-endian 64-bit limbs).
    #[inline]
    #[must_use]
    pub fn pow(&self, exp: impl AsRef<[u64]>) -> FieldElement {
        match self {
            FieldElement::Bn254(value) => FieldElement::Bn254(value.pow(exp)),
            FieldElement::Goldilocks(value) => FieldElement::Goldilocks(value.pow(exp)),
        }
    }

    /// The canonical integer this element denotes, zero-extended to four little-endian 64-bit
    /// limbs.
    #[inline]
    #[must_use]
    pub fn into_bigint(&self) -> BigInt<4> {
        match self {
            FieldElement::Bn254(value) => value.into_bigint(),
            FieldElement::Goldilocks(value) => BigInt([value.value(), 0, 0, 0]),
        }
    }

    /// The element's storage cells: its Montgomery-form limbs, as many as the field's
    /// `MavrosField::STORAGE_CELLS`.
    ///
    /// This is the representation the backends emit into bytecode words and the LLVM field struct.
    /// The canonical integer is [`FieldElement::into_bigint`].
    #[inline]
    #[must_use]
    pub fn storage_cells(&self) -> &[u64] {
        match self {
            FieldElement::Bn254(value) => &value.0.0,
            FieldElement::Goldilocks(value) => value.cells(),
        }
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, Field, UniformRand};
    use ark_std::rand::RngCore;

    use super::*;
    use crate::field::mavros_field::MavrosField;

    type Fr = ark_bn254::Fr;

    fn samples() -> Vec<Fr> {
        let mut rng = ark_std::test_rng();
        (0..256).map(|_| Fr::rand(&mut rng)).collect()
    }

    fn goldilocks_samples() -> Vec<Goldilocks> {
        let mut rng = ark_std::test_rng();
        let mut values: Vec<Goldilocks> =
            (0..256).map(|_| Goldilocks::from(rng.next_u64())).collect();
        values.extend([Goldilocks::ZERO, Goldilocks::ONE, -Goldilocks::ONE]);
        values
    }

    #[test]
    fn a_wrapped_element_belongs_to_its_field() {
        for f in samples() {
            assert_eq!(FieldElement::from(f), FieldElement::Bn254(f));
            assert_eq!(FieldElement::from(f).field_id(), FieldId::Bn254);
        }
    }

    #[test]
    fn storage_cells_are_the_montgomery_limbs() {
        for f in samples() {
            assert_eq!(FieldElement::from(f).storage_cells(), &f.0.0);
        }
        for g in goldilocks_samples() {
            let fe = FieldElement::from(g);
            assert_eq!(fe.storage_cells(), &[g.storage_cell0()]);
            assert_eq!(fe.field_id(), FieldId::Goldilocks);
        }
    }

    #[test]
    fn canonical_bigint_is_the_integer() {
        for f in samples() {
            assert_eq!(FieldElement::from(f).into_bigint(), f.into_bigint());
        }
        for g in goldilocks_samples() {
            let limbs = FieldElement::from(g).into_bigint().0;
            assert_eq!(limbs, [g.to_canonical_limbs()[0], 0, 0, 0]);
        }
    }

    #[test]
    fn arithmetic_matches_backing_all_operand_combos() {
        fn check<F>(a: F, b: F)
        where
            F: MavrosField + Div<Output = F> + Into<FieldElement>,
        {
            let (fa, fb): (FieldElement, FieldElement) = (a.into(), b.into());
            let wrap = |value: F| -> FieldElement { value.into() };

            // All four operand combinations of each operator must match the backing field.
            for (op_wrapped, op_raw) in [
                ((fa + fb, fa + &fb, &fa + fb, &fa + &fb), a + b),
                ((fa - fb, fa - &fb, &fa - fb, &fa - &fb), a - b),
                ((fa * fb, fa * &fb, &fa * fb, &fa * &fb), a * b),
            ] {
                let expected = wrap(op_raw);
                assert_eq!(op_wrapped.0, expected);
                assert_eq!(op_wrapped.1, expected);
                assert_eq!(op_wrapped.2, expected);
                assert_eq!(op_wrapped.3, expected);
            }

            if b != F::ZERO {
                let expected = wrap(a / b);
                assert_eq!(fa / fb, expected);
                assert_eq!(fa / &fb, expected);
                assert_eq!(&fa / fb, expected);
                assert_eq!(&fa / &fb, expected);
                // a / b == a * b^-1
                assert_eq!(fa / fb, fa * fb.inverse().unwrap());
            }

            assert_eq!((-fa), wrap(-a));
            assert_eq!((-&fa), wrap(-a));

            let mut acc = fa;
            acc += fb;
            assert_eq!(acc, wrap(a + b));
            let mut acc = fa;
            acc -= &fb;
            assert_eq!(acc, wrap(a - b));
            let mut acc = fa;
            acc *= fb;
            assert_eq!(acc, wrap(a * b));
        }

        let mut rng = ark_std::test_rng();
        for _ in 0..1024 {
            check(Fr::rand(&mut rng), Fr::rand(&mut rng));
            check(
                Goldilocks::from(rng.next_u64()),
                Goldilocks::from(rng.next_u64()),
            );
        }
    }

    #[test]
    #[should_panic(expected = "two fields (Bn254 and Goldilocks)")]
    fn mixed_field_arithmetic_panics() {
        let _ = FieldElement::from(Fr::from(1u64)) + FieldElement::from(Goldilocks::from(1u64));
    }

    #[test]
    #[should_panic(expected = "two fields (Goldilocks and Bn254)")]
    fn mixed_field_comparison_panics() {
        let _ = FieldElement::from(Goldilocks::from(1u64)) < FieldElement::from(Fr::from(1u64));
    }

    #[test]
    #[should_panic(expected = "two fields (Bn254 and Goldilocks)")]
    fn mixed_field_equality_panics() {
        let _ = FieldElement::from(Fr::from(1u64)) == FieldElement::from(Goldilocks::from(1u64));
    }

    #[test]
    fn identities_match_backing() {
        let bn254_zero = FieldElement::from(<Fr as AdditiveGroup>::ZERO);
        let bn254_one = FieldElement::from(<Fr as Field>::ONE);
        let goldilocks_zero = FieldElement::from(Goldilocks::ZERO);
        let goldilocks_one = FieldElement::from(Goldilocks::ONE);
        for (zero, one) in [(bn254_zero, bn254_one), (goldilocks_zero, goldilocks_one)] {
            assert!(zero.is_zero());
            assert!(!one.is_zero());
            assert!(one.is_one());
            assert!(!zero.is_one());
            assert_eq!(zero.inverse(), None);
            assert_eq!(one.inverse(), Some(one));
        }
    }

    #[test]
    fn pow_and_inverse_match_backing() {
        for f in samples() {
            let fe = FieldElement::from(f);
            for e in [0u64, 1, 2, 5, 32, 253] {
                assert_eq!(fe.pow([e]), FieldElement::from(f.pow([e])));
            }
            assert_eq!(fe.inverse(), Field::inverse(&f).map(FieldElement::from));
        }
        for g in goldilocks_samples() {
            let fe = FieldElement::from(g);
            for e in [0u64, 1, 2, 5, 32, 63, 64] {
                assert_eq!(fe.pow([e]), FieldElement::from(g.pow([e])));
            }
            assert_eq!(fe.inverse(), g.inverse().map(FieldElement::from));
        }
    }

    #[test]
    fn hash_ord_eq_consistent_with_backing() {
        use std::{
            collections::BTreeSet,
            hash::{BuildHasher, BuildHasherDefault, DefaultHasher},
        };

        let raw = samples();
        let wrapped: BTreeSet<FieldElement> = raw.iter().copied().map(FieldElement::from).collect();
        let raw_set: BTreeSet<Fr> = raw.iter().copied().collect();

        // Identical sorted iteration order — the property CSE interning depends on.
        let wrapped_sorted: Vec<Fr> = wrapped.iter().map(|f| Fr::from_element(*f)).collect();
        let raw_sorted: Vec<Fr> = raw_set.iter().copied().collect();
        assert_eq!(wrapped_sorted, raw_sorted);

        // A bn254 element hashes exactly as its raw value does.
        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        for f in &raw {
            assert_eq!(hasher.hash_one(FieldElement::from(*f)), hasher.hash_one(*f));
        }

        for a in &raw {
            for b in &raw {
                let (fa, fb) = (FieldElement::from(*a), FieldElement::from(*b));
                assert_eq!(fa == fb, a == b);
                assert_eq!(fa.cmp(&fb), a.cmp(b));
                assert_eq!(fa.partial_cmp(&fb), a.partial_cmp(b));
            }
        }

        let goldilocks = goldilocks_samples();
        for a in &goldilocks {
            for b in &goldilocks {
                let (fa, fb) = (FieldElement::from(*a), FieldElement::from(*b));
                assert_eq!(fa == fb, a == b);
                assert_eq!(fa.cmp(&fb), a.cmp(b));
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
        for g in goldilocks_samples() {
            let fe = FieldElement::from(g);
            assert_eq!(format!("{fe}"), format!("{g}"));
            assert_eq!(format!("{fe:?}"), format!("{g:?}"));
        }
    }
}
