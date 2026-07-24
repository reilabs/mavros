//! The operations that Mavros requires of a given field.
//!
//! Support for multiple fields in the VM is achieved by each field having its own set of named
//! opcodes that are emitted into the bytecode by the compiler for a concrete field (e.g. rather
//! than `FieldAdd` we have `Bn254FieldAdd`). This ensures that a single VM binary can be used with
//! _any_ supported field, while also avoiding the need to perform potentially-expensive runtime
//! field selection.
//!
//! In order to ensure that the opcode generator can uniformly generate that per-field opcode set,
//! the [`MavrosField`] trait exists to ensure the operations exist.

use ark_ff::{AdditiveGroup, BigInt, Field as _};

// FIELD IDENTIFIER
// ================================================================================================

/// Identifies one of Mavros' supported fields.
///
/// The enum discriminant *is* the wire encoding: encoding is a direct `id as u32` (see
/// [`FieldId::code`]). Decoding must go through the checked [`FieldId::from_code`] — casting an
/// untrusted `u32` straight into the enum would be undefined behaviour for an unknown value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum FieldId {
    Bn254 = 0,
}

impl FieldId {
    /// The wire encoding — the enum discriminant.
    #[must_use]
    pub fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a [`FieldId`] from its wire encoding, or `None` if the value is not a known field.
    #[must_use]
    pub fn from_code(code: u32) -> Option<FieldId> {
        match code {
            0 => Some(FieldId::Bn254),
            _ => None,
        }
    }
}

// FIELD TRAIT
// ================================================================================================

/// The operations supported by a given field.
///
/// The plan is to have a set of field opcodes per supported field, each set of which will dispatch
/// to a backing implementation of this trait. This ensures that the VM can dispatch directly on
/// operations on a concrete field, rather than paying a penalty to check the field modulus every
/// time.
///
/// The associated [`MavrosField::STORAGE_CELLS`] is the number of 64-bit cells one element of this
/// field occupies in the VM frame / bytecode operand / constant pool. The `*_frame_cells` and
/// `*_limbs_le` methods move a value through that raw storage form and the arithmetic methods are
/// what the concrete field opcodes call.
pub trait MavrosField: Copy + PartialEq {
    /// Number of 64-bit storage cells one element occupies in the raw representation.
    const STORAGE_CELLS: usize;

    /// Which field this is.
    const FIELD_ID: FieldId;

    /// The by-value raw storage limbs of one element (bytecode operand form): a fixed-size array
    /// whose length is this field's [`MavrosField::STORAGE_CELLS`].
    type Limbs: AsRef<[u64]> + Copy;

    /// The additive identity.
    fn zero() -> Self;

    /// The multiplicative identity.
    fn one() -> Self;

    /// Field element addition.
    fn add(self, rhs: Self) -> Self;

    /// Field element subtraction.
    fn sub(self, rhs: Self) -> Self;

    /// Field element multiplication.
    fn mul(self, rhs: Self) -> Self;

    /// Field element division.
    fn div(self, rhs: Self) -> Self;

    /// The multiplicative inverse, or `None` for zero.
    fn inverse(self) -> Option<Self>;

    /// Reconstructs an element from its raw storage cells (as read from a VM frame).
    fn from_frame_cells(cells: &[u64]) -> Self;

    /// Writes an element's raw storage cells (as written into a VM frame).
    fn to_frame_cells(self, out: &mut [u64]);

    /// Reconstructs from raw little-endian storage limbs (bytecode operand form).
    fn from_limbs_le(limbs: &[u64]) -> Self;

    /// The raw little-endian storage limbs (bytecode operand form), by value.
    ///
    /// Returns the per-field [`MavrosField::Limbs`] array, so the trait carries no bn254-specific
    /// width while callers keep an owned array rather than pre-allocating a buffer; read it as a
    /// slice via `AsRef<[u64]>`.
    fn to_limbs_le(self) -> Self::Limbs;
}

// BN254 UNDERLYING FIELD REP
// ================================================================================================

/// The bn254 scalar field.
///
/// This wraps the raw field verbatim (Montgomery storage form), matching the representation the VM
/// frame, opcode operands, and constant pool use today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bn254Field(ark_bn254::Fr);

impl Bn254Field {
    /// Wraps a raw backing field element.
    #[must_use]
    pub fn from_ark(value: ark_bn254::Fr) -> Bn254Field {
        Bn254Field(value)
    }

    /// The raw backing field element.
    #[must_use]
    pub fn to_ark(self) -> ark_bn254::Fr {
        self.0
    }
}

impl MavrosField for Bn254Field {
    const STORAGE_CELLS: usize = 4;
    const FIELD_ID: FieldId = FieldId::Bn254;
    type Limbs = [u64; Self::STORAGE_CELLS];

    fn zero() -> Self {
        Bn254Field(<ark_bn254::Fr as AdditiveGroup>::ZERO)
    }

    fn one() -> Self {
        Bn254Field(<ark_bn254::Fr as ark_ff::Field>::ONE)
    }

    fn add(self, rhs: Self) -> Self {
        Bn254Field(self.0 + rhs.0)
    }

    fn sub(self, rhs: Self) -> Self {
        Bn254Field(self.0 - rhs.0)
    }

    fn mul(self, rhs: Self) -> Self {
        Bn254Field(self.0 * rhs.0)
    }

    fn div(self, rhs: Self) -> Self {
        Bn254Field(self.0 / rhs.0)
    }

    fn inverse(self) -> Option<Self> {
        self.0.inverse().map(Bn254Field)
    }

    fn from_frame_cells(cells: &[u64]) -> Self {
        debug_assert!(cells.len() >= Self::STORAGE_CELLS);

        // `new_unchecked` performs no reduction: `cells` is taken as an already-canonical
        // Montgomery encoding, which holds because it only ever comes from `to_frame_cells`/the VM
        // frame.
        Bn254Field(ark_bn254::Fr::new_unchecked(BigInt::new([
            cells[0], cells[1], cells[2], cells[3],
        ])))
    }

    fn to_frame_cells(self, out: &mut [u64]) {
        out[..Self::STORAGE_CELLS].copy_from_slice(&self.0.0.0);
    }

    fn from_limbs_le(limbs: &[u64]) -> Self {
        Self::from_frame_cells(limbs)
    }

    fn to_limbs_le(self) -> [u64; 4] {
        self.0.0.0
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;

    use super::*;

    #[test]
    fn field_id_code_round_trips() {
        assert_eq!(
            FieldId::from_code(FieldId::Bn254.code()),
            Some(FieldId::Bn254)
        );
        assert_eq!(FieldId::from_code(1), None);
    }

    #[test]
    fn bn254_storage_is_four_cells() {
        assert_eq!(Bn254Field::STORAGE_CELLS, 4);
        assert_eq!(Bn254Field::FIELD_ID, FieldId::Bn254);
    }

    #[test]
    fn bn254_ops_match_backing() {
        let mut rng = ark_std::test_rng();
        for _ in 0..256 {
            let a = ark_bn254::Fr::rand(&mut rng);
            let b = ark_bn254::Fr::rand(&mut rng);
            let (fa, fb) = (Bn254Field::from_ark(a), Bn254Field::from_ark(b));
            assert_eq!(fa.add(fb).to_ark(), a + b);
            assert_eq!(fa.sub(fb).to_ark(), a - b);
            assert_eq!(fa.mul(fb).to_ark(), a * b);
            assert_eq!(
                fa.inverse().map(Bn254Field::to_ark),
                ark_ff::Field::inverse(&a)
            );
        }
        assert_eq!(
            Bn254Field::zero().to_ark(),
            <ark_bn254::Fr as AdditiveGroup>::ZERO
        );
        assert_eq!(
            Bn254Field::one().to_ark(),
            <ark_bn254::Fr as ark_ff::Field>::ONE
        );
    }

    #[test]
    fn bn254_frame_cell_round_trip_is_montgomery() {
        let mut rng = ark_std::test_rng();
        for _ in 0..256 {
            let a = ark_bn254::Fr::rand(&mut rng);
            let fa = Bn254Field::from_ark(a);
            let mut cells = [0u64; 4];
            fa.to_frame_cells(&mut cells);
            // Raw Montgomery limbs, matching the VM frame representation.
            assert_eq!(cells, a.0.0);
            assert_eq!(Bn254Field::from_frame_cells(&cells), fa);
            let limbs = fa.to_limbs_le();
            assert_eq!(limbs, a.0.0);
            // The associated `Limbs` array must stay consistent with `STORAGE_CELLS`.
            assert_eq!(limbs.as_ref().len(), Bn254Field::STORAGE_CELLS);
            assert_eq!(Bn254Field::from_limbs_le(&limbs), fa);
        }
    }
}
