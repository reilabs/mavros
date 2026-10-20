//! The operations that Mavros requires of a given field.
//!
//! Support for multiple fields in the VM is achieved by each field having its own set of named
//! opcodes that are emitted into the bytecode by the compiler for a concrete field (e.g. bn254's
//! copy of `add_field` is `bn254_add_field`). This ensures that a single VM binary can be used with
//! _any_ supported field, while also avoiding the need to perform potentially-expensive runtime
//! field selection.
//!
//! The [`MavrosField`] trait is used to make the opcode handlers compile-time generic, and is
//! implemented directly for the choice of raw backing field type.
//!
//! A field element has two distinct limb forms, and mixing them up can cause silent value
//! corruption.
//!
//! - **Storage cells** (`*_frame_cells`, [`MavrosField::storage_cell0`]) are the element's
//!   in-memory representation (Montgomery form for every supported field). This is what VM
//!   frames, tapes, bytecode operands and the constant pool hold.
//! - **Canonical limbs** (`*_canonical_limbs`) are the little-endian limbs of the integer in
//!   `0..modulus` that the element denotes. Ordering, casts to integers, bit/byte decompositions
//!   and table-row addressing are defined on this form.

use std::{
    cmp::Ordering,
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ff::{AdditiveGroup, BigInt, PrimeField};
use serde::{Serialize, de::DeserializeOwned};

use crate::field::element::FieldElement;

// FIELD IDENTIFIER
// ================================================================================================

/// Identifies one of Mavros' supported fields.
///
/// A VM program binary names its field in its header ([`FieldId::code`]), and the VM refuses to run
/// it over another; its field opcodes name the field too. In the compiler it selects per-field
/// behaviour through exhaustive matches, so a new variant is a compile error at every per-field
/// site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldId {
    Bn254,
    Goldilocks,
}

impl FieldId {
    /// The word naming this field in a VM program header. No field is `0`, so a zeroed word names
    /// none.
    #[must_use]
    pub const fn code(self) -> u64 {
        match self {
            FieldId::Bn254 => 1,
            FieldId::Goldilocks => 2,
        }
    }

    /// The field a program header word names, or `None` for a word that names none.
    #[must_use]
    pub const fn from_code(code: u64) -> Option<FieldId> {
        match code {
            1 => Some(FieldId::Bn254),
            2 => Some(FieldId::Goldilocks),
            _ => None,
        }
    }
}

// FIELD TRAIT
// ================================================================================================

/// The operations the VM performs on the elements of one field.
///
/// `From<u64>` and `From<u128>` should reduce modulo the field's modulus, so an integer wider than
/// the field wraps into it.
///
/// # Safety
///
/// The VM reinterprets frame and tape memory (`u64` cells) as `Self` through pointer casts, and
/// keeps plain `u64` counters and flags in cell 0 of field slots on its tapes. An implementation
/// of this trait must guarantee that:
///
/// - `size_of::<Self>() == STORAGE_CELLS * 8`, `align_of::<Self>() <= align_of::<u64>()`, and
///   `STORAGE_CELLS >= 1`.
/// - The bytes of a value are exactly the cells written by [`MavrosField::to_frame_cells`], so a
///   cast read agrees with [`MavrosField::from_frame_cells`] and a write with `to_frame_cells`.
/// - Every pattern of `STORAGE_CELLS` cells is a valid `Self`, including one that denotes no
///   element, and copying or comparing one is not undefined behavior: a slot holding a punned `u64`
///   is still handled as `Self`.
/// - [`MavrosField::ZERO`] is stored as all-zero cells, so zero-filled memory reads as `ZERO`.
pub unsafe trait MavrosField:
    Copy
    + Eq
    + Debug
    + Display
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + From<u64>
    + From<u128>
{
    /// The field this type implements.
    const ID: FieldId;

    /// Number of 64-bit storage cells one element occupies.
    const STORAGE_CELLS: usize;

    /// Number of 64-bit limbs in the canonical integer form, enough to hold `modulus - 1`.
    const CANONICAL_LIMBS: usize;

    /// The canonical limbs of one element: a fixed-size array of
    /// [`MavrosField::CANONICAL_LIMBS`] limbs, read as a slice via `AsRef<[u64]>`.
    type Limbs: AsRef<[u64]> + Copy + Serialize + DeserializeOwned;

    /// The additive identity.
    const ZERO: Self;

    /// The multiplicative identity.
    const ONE: Self;

    /// The multiplicative inverse, or `None` for zero.
    fn inverse(&self) -> Option<Self>;

    /// The raw element a compiler [`FieldElement`] of this field wraps.
    ///
    /// # Panics
    ///
    /// If `value` belongs to another field.
    fn from_element(value: FieldElement) -> Self;

    /// This element as a compiler [`FieldElement`].
    fn to_element(self) -> FieldElement;

    /// Reconstructs an element from its storage cells, keeping them as given: the WASM host reads a
    /// slot holding a punned `u64` through this, and [`MavrosField::storage_cell0`] of the result
    /// must be that `u64`.
    ///
    /// `cells` must hold at least [`MavrosField::STORAGE_CELLS`] cells. A pattern that
    /// [`MavrosField::to_frame_cells`] did not write need not denote an element.
    fn from_frame_cells(cells: &[u64]) -> Self;

    /// Writes an element's [`MavrosField::STORAGE_CELLS`] storage cells to the front of `out`.
    fn to_frame_cells(self, out: &mut [u64]);

    /// The element's first storage cell.
    ///
    /// The VM keeps a `u64` punned into a field slot in this cell. For a field stored in Montgomery
    /// form it differs from the low canonical limb.
    fn storage_cell0(self) -> u64;

    /// The canonical little-endian limbs of the integer this element denotes.
    fn to_canonical_limbs(self) -> Self::Limbs;

    /// The element denoting the little-endian integer `limbs`, or `None` if it is not below the
    /// modulus. Fewer than [`MavrosField::CANONICAL_LIMBS`] limbs are zero-extended.
    ///
    /// # Panics
    ///
    /// If `limbs` holds more than [`MavrosField::CANONICAL_LIMBS`] limbs.
    fn from_canonical_limbs(limbs: &[u64]) -> Option<Self>;

    /// Orders two elements by the integers they denote.
    fn canonical_cmp(self, other: Self) -> Ordering {
        let (a, b) = (self.to_canonical_limbs(), other.to_canonical_limbs());
        a.as_ref().iter().rev().cmp(b.as_ref().iter().rev())
    }
}

/// Panics for a [`MavrosField::from_element`] given an element of another field.
#[cold]
#[track_caller]
pub(crate) fn foreign_element(value: FieldElement, field: FieldId) -> ! {
    panic!(
        "ICE: a {:?} element where a {field:?} one belongs",
        value.field_id()
    )
}

// BN254
// ================================================================================================

// ark gives `Fp` no `repr`, so the language does not fix `Fr`'s layout, but we do here. `Fr` holds
// a `[u64; 4]` and a `PhantomData`, so at 32 bytes the array is the whole value, at offset 0. We
// rely on this, so these asserts ensure the compilation fails loudly should it ever change.
const _: () =
    assert!(size_of::<ark_bn254::Fr>() == 32 && align_of::<ark_bn254::Fr>() <= align_of::<u64>());
const _: () = assert!(matches!(
    <ark_bn254::Fr as AdditiveGroup>::ZERO.0.0,
    [0, 0, 0, 0]
));

/// The bn254 scalar field, stored as its four Montgomery-form limbs.
///
/// # Safety
///
/// The assertions above make `Fr` exactly its limb array, which `to_frame_cells` writes and whose
/// every pattern is a valid `Fr` (plain integers, no niche), and make `ZERO` all-zero cells.
unsafe impl MavrosField for ark_bn254::Fr {
    const ID: FieldId = FieldId::Bn254;
    const STORAGE_CELLS: usize = 4;
    const CANONICAL_LIMBS: usize = 4;
    type Limbs = [u64; 4];
    const ZERO: Self = <ark_bn254::Fr as AdditiveGroup>::ZERO;
    const ONE: Self = <ark_bn254::Fr as ark_ff::Field>::ONE;

    fn inverse(&self) -> Option<Self> {
        ark_ff::Field::inverse(self)
    }

    #[track_caller]
    fn from_element(value: FieldElement) -> Self {
        match value {
            FieldElement::Bn254(value) => value,
            _ => foreign_element(value, FieldId::Bn254),
        }
    }

    fn to_element(self) -> FieldElement {
        FieldElement::Bn254(self)
    }

    fn from_frame_cells(cells: &[u64]) -> Self {
        // `new_unchecked` neither converts nor reduces, so a punned `u64` or a pattern past the
        // modulus is kept as given.
        ark_bn254::Fr::new_unchecked(BigInt::new([cells[0], cells[1], cells[2], cells[3]]))
    }

    fn to_frame_cells(self, out: &mut [u64]) {
        out[..Self::STORAGE_CELLS].copy_from_slice(&self.0.0);
    }

    fn storage_cell0(self) -> u64 {
        self.0.0[0]
    }

    fn to_canonical_limbs(self) -> [u64; 4] {
        self.into_bigint().0
    }

    fn from_canonical_limbs(limbs: &[u64]) -> Option<Self> {
        assert!(
            limbs.len() <= Self::CANONICAL_LIMBS,
            "{} limbs do not fit a bn254 element's {} canonical limbs",
            limbs.len(),
            Self::CANONICAL_LIMBS
        );
        let mut padded = [0u64; 4];
        padded[..limbs.len()].copy_from_slice(limbs);
        ark_bn254::Fr::from_bigint(BigInt(padded))
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use ark_ff::UniformRand;

    use super::*;

    type Fr = ark_bn254::Fr;

    fn samples() -> Vec<Fr> {
        let mut rng = ark_std::test_rng();
        let mut values: Vec<Fr> = (0..256).map(|_| Fr::rand(&mut rng)).collect();
        values.extend([
            <Fr as MavrosField>::ZERO,
            <Fr as MavrosField>::ONE,
            -<Fr as MavrosField>::ONE,
            Fr::from(u64::MAX),
            Fr::from(u128::MAX),
        ]);
        values
    }

    #[test]
    fn bn254_layout_is_its_storage_cells() {
        assert_eq!(size_of::<Fr>(), <Fr as MavrosField>::STORAGE_CELLS * 8);
        assert!(align_of::<Fr>() <= align_of::<u64>());
        for value in samples() {
            let mut cells = [0u64; 4];
            value.to_frame_cells(&mut cells);
            // SAFETY: the size was checked above, and `cells` is `u64`-aligned.
            let punned = unsafe { *(cells.as_ptr() as *const Fr) };
            assert_eq!(punned, value);
            // SAFETY: as above, reading the element's bytes back as cells.
            let bytes = unsafe { *(&value as *const Fr as *const [u64; 4]) };
            assert_eq!(bytes, cells);
        }
    }

    #[test]
    fn bn254_storage_cells_are_montgomery() {
        for value in samples() {
            let mut cells = [0u64; 4];
            value.to_frame_cells(&mut cells);
            assert_eq!(cells, value.0.0);
            assert_eq!(<Fr as MavrosField>::from_frame_cells(&cells), value);
            assert_eq!(value.storage_cell0(), value.0.0[0]);
        }
        // Montgomery form is visible: `1` is not stored as the integer 1.
        assert_ne!(<Fr as MavrosField>::ONE.storage_cell0(), 1);
        // A punned `u64`, and cells past the modulus, are kept as given.
        for cells in [[7, 0, 0, 0], [u64::MAX; 4]] {
            assert_eq!(
                <Fr as MavrosField>::from_frame_cells(&cells).storage_cell0(),
                cells[0]
            );
        }
    }

    #[test]
    fn bn254_canonical_limbs_are_the_integer() {
        for value in samples() {
            let limbs = value.to_canonical_limbs();
            assert_eq!(limbs, value.into_bigint().0);
            assert_eq!(limbs.len(), <Fr as MavrosField>::CANONICAL_LIMBS);
            assert_eq!(
                <Fr as MavrosField>::from_canonical_limbs(&limbs),
                Some(value)
            );
        }
        assert_eq!(Fr::from(7u64).to_canonical_limbs(), [7, 0, 0, 0]);
        // Short inputs are zero-extended.
        assert_eq!(
            <Fr as MavrosField>::from_canonical_limbs(&[5, 1]),
            Some(Fr::from((1u128 << 64) | 5))
        );
        assert_eq!(
            <Fr as MavrosField>::from_canonical_limbs(&[]),
            Some(<Fr as MavrosField>::ZERO)
        );
        // The modulus itself has no element.
        assert_eq!(
            <Fr as MavrosField>::from_canonical_limbs(&Fr::MODULUS.0),
            None
        );
    }

    #[test]
    fn bn254_elements_round_trip_through_field_element() {
        assert_eq!(<Fr as MavrosField>::ID, FieldId::Bn254);
        for value in samples() {
            let element = value.to_element();
            assert_eq!(element, FieldElement::Bn254(value));
            assert_eq!(<Fr as MavrosField>::from_element(element), value);
        }
    }

    #[test]
    #[should_panic(expected = "a Goldilocks element where a Bn254 one belongs")]
    fn bn254_from_element_refuses_a_goldilocks_element() {
        let element = FieldElement::Goldilocks(crate::field::goldilocks::Goldilocks::from(1u64));
        let _ = <Fr as MavrosField>::from_element(element);
    }

    #[test]
    #[should_panic(expected = "do not fit")]
    fn bn254_from_canonical_limbs_rejects_too_many_limbs() {
        let _ = <Fr as MavrosField>::from_canonical_limbs(&[0; 5]);
    }

    #[test]
    fn bn254_canonical_cmp_matches_integer_order() {
        let values = samples();
        for a in &values {
            for b in values.iter().take(32) {
                assert_eq!(a.canonical_cmp(*b), a.into_bigint().cmp(&b.into_bigint()));
            }
        }
        // The samples include pairs whose storage cells order differently from their integers, so
        // the loop above would catch a comparison of the cells.
        let storage_cmp = |a: &Fr, b: &Fr| a.0.0.iter().rev().cmp(b.0.0.iter().rev());
        assert!(values.iter().any(|a| {
            values
                .iter()
                .any(|b| storage_cmp(a, b) != a.into_bigint().cmp(&b.into_bigint()))
        }));
        let (two, three) = (Fr::from(2u64), Fr::from(3u64));
        assert_eq!(two.canonical_cmp(three), Ordering::Less);
        assert_eq!((-two).canonical_cmp(two), Ordering::Greater);
    }

    #[test]
    fn bn254_arithmetic_matches_backing() {
        let values = samples();
        for pair in values.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert_eq!(MavrosField::inverse(&a), ark_ff::Field::inverse(&a));
            if let Some(inverse) = MavrosField::inverse(&b) {
                assert_eq!(a * inverse, a / b);
            }
        }
        assert_eq!(MavrosField::inverse(&<Fr as MavrosField>::ZERO), None);
        assert_eq!(<Fr as MavrosField>::ZERO, <Fr as AdditiveGroup>::ZERO);
        assert_eq!(<Fr as MavrosField>::ONE, <Fr as ark_ff::Field>::ONE);
    }

    /// `fake_challenges` seeds its challenges with `F::from` of this integer, which for bn254 must
    /// be the element the integer names.
    #[test]
    fn bn254_fake_challenge_seed_converts_exactly() {
        let seed: u128 = 18765435241434657586764563434227903;
        let expected = Fr::from_bigint(BigInt::from_str(&seed.to_string()).unwrap()).unwrap();
        assert_eq!(Fr::from(seed), expected);
    }
}
