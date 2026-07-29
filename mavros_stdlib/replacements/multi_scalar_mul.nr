// Pure-Noir replacement for the `multi_scalar_mul` foreign function.
//
// The embedded curve for BN254 is Grumpkin, with equation y^2 = x^3 - 17.
// Scalars are represented by two 128-bit little-endian limbs. The implementation
// uses a binary Straus MSM in Jacobian coordinates: one doubling per scalar bit,
// followed by one conditional addition per input point. Keeping the accumulator
// projective avoids an inversion for every group operation.

use crate::embedded_curve_ops::{EmbeddedCurvePoint, EmbeddedCurveScalar};

global GRUMPKIN_SCALAR_MODULUS_LO: u128 = 0x97816a916871ca8d3c208c16d87cfd47;
global GRUMPKIN_SCALAR_MODULUS_HI: u128 = 0x30644e72e131a029b85045b68181585d;

struct JacobianPoint {
    x: Field,
    y: Field,
    z: Field,
}

fn infinity() -> JacobianPoint {
    // Any point with z = 0 represents infinity. Using y = 1 gives the
    // conventional Jacobian representation and keeps doubling well-defined.
    JacobianPoint { x: 0, y: 1, z: 0 }
}

fn select_point(condition: bool, if_true: JacobianPoint, if_false: JacobianPoint) -> JacobianPoint {
    if condition { if_true } else { if_false }
}

fn from_affine(point: EmbeddedCurvePoint) -> JacobianPoint {
    let is_infinite = point.is_infinite();
    if is_infinite {
        infinity()
    } else {
        JacobianPoint { x: point.x, y: point.y, z: 1 }
    }
}

// Jacobian doubling for a short-Weierstrass curve with a = 0.
fn double(point: JacobianPoint) -> JacobianPoint {
    let a = point.x * point.x;
    let b = point.y * point.y;
    let c = b * b;
    let x_plus_b = point.x + b;
    let d = 2 * (x_plus_b * x_plus_b - a - c);
    let e = 3 * a;
    let f = e * e;
    let x = f - 2 * d;
    let y = e * (d - x) - 8 * c;
    let z = 2 * point.y * point.z;
    JacobianPoint { x, y, z }
}

// Complete wrapper around the standard Jacobian addition formula. The formula
// itself handles distinct finite points; explicit selections cover equality,
// inverses, and either input being infinity.
fn add(lhs: JacobianPoint, rhs: JacobianPoint) -> JacobianPoint {
    let lhs_z2 = lhs.z * lhs.z;
    let rhs_z2 = rhs.z * rhs.z;
    let lhs_u = lhs.x * rhs_z2;
    let rhs_u = rhs.x * lhs_z2;
    let lhs_s = lhs.y * rhs.z * rhs_z2;
    let rhs_s = rhs.y * lhs.z * lhs_z2;

    let h = rhs_u - lhs_u;
    let i = (2 * h) * (2 * h);
    let j = h * i;
    let r = 2 * (rhs_s - lhs_s);
    let v = lhs_u * i;
    let x = r * r - j - 2 * v;
    let y = r * (v - x) - 2 * lhs_s * j;
    let z_sum = lhs.z + rhs.z;
    let z = (z_sum * z_sum - lhs_z2 - rhs_z2) * h;
    let generic_sum = JacobianPoint { x, y, z };

    let lhs_infinite = lhs.z == 0;
    let rhs_infinite = rhs.z == 0;
    let same_point = (lhs_u == rhs_u) & (lhs_s == rhs_s);
    let finite_sum = select_point(same_point, double(lhs), generic_sum);
    let with_rhs_infinity = select_point(rhs_infinite, lhs, finite_sum);
    select_point(lhs_infinite, rhs, with_rhs_infinity)
}

fn to_affine(point: JacobianPoint) -> EmbeddedCurvePoint {
    let is_infinite = point.z == 0;

    // Adding one only for infinity makes the denominator non-zero without
    // changing finite points. This avoids ever asking witgen to invert zero.
    let safe_z = point.z + is_infinite as Field;
    let z_inverse = 1 / safe_z;
    let z_inverse_2 = z_inverse * z_inverse;
    let x = point.x * z_inverse_2;
    let y = point.y * z_inverse_2 * z_inverse;

    if is_infinite {
        EmbeddedCurvePoint::point_at_infinity()
    } else {
        EmbeddedCurvePoint { x, y }
    }
}

fn scalar_is_valid(lo: u128, hi: u128) -> bool {
    (hi < GRUMPKIN_SCALAR_MODULUS_HI)
        | ((hi == GRUMPKIN_SCALAR_MODULUS_HI) & (lo < GRUMPKIN_SCALAR_MODULUS_LO))
}

unconstrained fn truncate_limb_hint(limb: Field) -> u128 {
    limb as u128
}

fn checked_limb(limb: Field) -> u128 {
    let value = if crate::runtime::is_unconstrained() {
        limb as u128
    } else {
        unsafe { truncate_limb_hint(limb) }
    };

    // The hint truncates. Equating its Field embedding with the original
    // limb proves that no high bits were discarded.
    assert_eq(limb, value as Field);
    value
}

pub fn multi_scalar_mul<let N: u32>(
    points: [EmbeddedCurvePoint; N],
    scalars: [EmbeddedCurveScalar; N],
    predicate: bool,
) -> [EmbeddedCurvePoint; 1] {
    let mut bases = [infinity(); N];
    let mut scalar_lo = [0 as u128; N];
    let mut scalar_hi = [0 as u128; N];

    for i in 0..N {
        // A false predicate suppresses input validation in the upstream
        // black-box contract and returns infinity.
        let point = if predicate {
            points[i]
        } else {
            EmbeddedCurvePoint::point_at_infinity()
        };
        let lo = if predicate { scalars[i].lo } else { 0 };
        let hi = if predicate { scalars[i].hi } else { 0 };

        let point_is_valid =
            point.is_infinite() | (point.y * point.y == point.x * point.x * point.x - 17);
        assert(point_is_valid);

        scalar_lo[i] = checked_limb(lo);
        scalar_hi[i] = checked_limb(hi);
        assert(scalar_is_valid(scalar_lo[i], scalar_hi[i]));
        bases[i] = from_affine(point);
    }

    let mut accumulator = infinity();

    // Process the 256-bit scalars from most-significant to least-significant
    // bit. Sharing the doubling across all points is substantially smaller
    // than performing N independent scalar multiplications.
    for round in 0..256 {
        accumulator = double(accumulator);
        let bit_index = 255 - round;
        for i in 0..N {
            let bit = if bit_index < 128 {
                ((scalar_lo[i] >> bit_index as u128) & 1) == 1
            } else {
                ((scalar_hi[i] >> (bit_index - 128) as u128) & 1) == 1
            };
            let sum = add(accumulator, bases[i]);
            accumulator = select_point(bit, sum, accumulator);
        }
    }

    let result = to_affine(accumulator);
    if predicate {
        [result]
    } else {
        [EmbeddedCurvePoint::point_at_infinity()]
    }
}
