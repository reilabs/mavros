//! Helpers for the VM's tests that run one program over more than one field.

use mavros_artifacts::MavrosField;

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
