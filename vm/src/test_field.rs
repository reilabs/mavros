//! A second field for the VM's tests: the Goldilocks prime `2^64 - 2^32 + 1`.
//!
//! It exists to test the per-field opcode machinery at a storage width other than bn254's. The
//! Goldilocks field the VM will support is separate work.

use std::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ff::{AdditiveGroup, BigInt, Fp64, MontBackend, MontConfig, PrimeField};
use mavros_artifacts::MavrosField;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct GoldilocksConfig;

type Goldilocks = Fp64<MontBackend<GoldilocksConfig, 1>>;

/// The test field's element. A newtype only because the orphan rule forbids implementing
/// `MavrosField` for ark's `Fp64` here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct TestField(Goldilocks);

impl TestField {
    /// The modulus, `2^64 - 2^32 + 1`.
    pub const MODULUS: u64 = 0xffff_ffff_0000_0001;
}

macro_rules! ring_op {
    ($op:ident, $method:ident, $assign_op:ident, $assign_method:ident) => {
        impl $op for TestField {
            type Output = TestField;

            fn $method(self, rhs: TestField) -> TestField {
                TestField($op::$method(self.0, rhs.0))
            }
        }

        impl $assign_op for TestField {
            fn $assign_method(&mut self, rhs: TestField) {
                $assign_op::$assign_method(&mut self.0, rhs.0);
            }
        }
    };
}

ring_op!(Add, add, AddAssign, add_assign);
ring_op!(Sub, sub, SubAssign, sub_assign);
ring_op!(Mul, mul, MulAssign, mul_assign);

impl Neg for TestField {
    type Output = TestField;

    fn neg(self) -> TestField {
        TestField(-self.0)
    }
}

impl From<u64> for TestField {
    fn from(value: u64) -> TestField {
        TestField(Goldilocks::from(value))
    }
}

impl From<u128> for TestField {
    fn from(value: u128) -> TestField {
        TestField(Goldilocks::from(value))
    }
}

impl fmt::Display for TestField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

// `Fp64` has no `repr`, so these assertions fix the layout: at 8 bytes, its one Montgomery limb is
// the whole value.
const _: () = assert!(size_of::<TestField>() == 8 && align_of::<TestField>() <= align_of::<u64>());
const _: () = assert!(matches!(<Goldilocks as AdditiveGroup>::ZERO.0.0, [0]));

// SAFETY: `repr(transparent)` over `Fp64`, which the assertions above make exactly its one limb:
// `to_frame_cells` writes it, every pattern of it is a valid value, and `ZERO`'s is 0.
unsafe impl MavrosField for TestField {
    const STORAGE_CELLS: usize = 1;
    const CANONICAL_LIMBS: usize = 1;
    type Limbs = [u64; 1];
    const ZERO: TestField = TestField(<Goldilocks as AdditiveGroup>::ZERO);
    const ONE: TestField = TestField(<Goldilocks as ark_ff::Field>::ONE);

    fn inverse(&self) -> Option<TestField> {
        ark_ff::Field::inverse(&self.0).map(TestField)
    }

    fn from_frame_cells(cells: &[u64]) -> TestField {
        TestField(Goldilocks::new_unchecked(BigInt::new([cells[0]])))
    }

    fn to_frame_cells(self, out: &mut [u64]) {
        out[0] = self.0.0.0[0];
    }

    fn storage_cell0(self) -> u64 {
        self.0.0.0[0]
    }

    fn to_canonical_limbs(self) -> [u64; 1] {
        self.0.into_bigint().0
    }

    fn from_canonical_limbs(limbs: &[u64]) -> Option<TestField> {
        assert!(
            limbs.len() <= Self::CANONICAL_LIMBS,
            "{} limbs do not fit the test field's one canonical limb",
            limbs.len()
        );
        Goldilocks::from_bigint(BigInt([limbs.first().copied().unwrap_or(0)])).map(TestField)
    }
}

/// `value` as the small signed integer it stands for, so that results over different fields can be
/// compared.
pub fn signed<F: MavrosField>(value: F) -> i128 {
    let small = |value: F| {
        let limbs = value.to_canonical_limbs();
        let limbs = limbs.as_ref();
        (limbs[1..].iter().all(|&limb| limb == 0) && limbs[0] < 1 << 40).then_some(limbs[0] as i128)
    };
    small(value)
        .or_else(|| small(-value).map(|magnitude| -magnitude))
        .expect("these tests compute only small values")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_one_storage_cell() {
        assert_eq!(size_of::<TestField>(), 8);
        assert!(align_of::<TestField>() <= align_of::<u64>());
        for value in [
            TestField::from(0u64),
            TestField::from(7u64),
            -TestField::ONE,
        ] {
            let mut cells = [0u64; 1];
            value.to_frame_cells(&mut cells);
            // SAFETY: one `u64` cell is exactly a `TestField`, as checked above.
            assert_eq!(unsafe { *(cells.as_ptr() as *const TestField) }, value);
            assert_eq!(TestField::from_frame_cells(&cells), value);
            assert_eq!(value.storage_cell0(), cells[0]);
        }
    }

    #[test]
    fn storage_is_montgomery_and_limbs_are_canonical() {
        assert_ne!(TestField::ONE.storage_cell0(), 1);
        // A punned `u64`, and a cell past the modulus, are kept as given.
        for cell in [7, u64::MAX] {
            assert_eq!(TestField::from_frame_cells(&[cell]).storage_cell0(), cell);
        }
        assert_eq!(TestField::ONE.to_canonical_limbs(), [1]);
        let top = -TestField::ONE;
        assert_eq!(top.to_canonical_limbs(), [TestField::MODULUS - 1]);
        assert_eq!(
            TestField::from_canonical_limbs(&[TestField::MODULUS - 1]),
            Some(top)
        );
        assert_eq!(TestField::from_canonical_limbs(&[TestField::MODULUS]), None);
        assert_eq!(TestField::from_canonical_limbs(&[]), Some(TestField::ZERO));
        assert_eq!(top + TestField::ONE, TestField::ZERO);
        assert_eq!(
            TestField::from(u64::MAX).to_canonical_limbs(),
            [(1 << 32) - 2]
        );
    }
}
