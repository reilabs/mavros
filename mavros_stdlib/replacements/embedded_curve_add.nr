// Pure-Noir replacement for the `embedded_curve_add` foreign function.
//
// The complete Jacobian addition used by the MSM replacement handles distinct
// points, doubling, inverses, and either input being the point at infinity.

use crate::embedded_curve_ops::EmbeddedCurvePoint;

use super::multi_scalar_mul::{add_affine, point_is_valid};

pub fn embedded_curve_add(
    point1: EmbeddedCurvePoint,
    point2: EmbeddedCurvePoint,
    predicate: bool,
) -> [EmbeddedCurvePoint; 1] {
    // A false predicate suppresses input validation in the upstream black-box
    // contract. Masking both points also keeps the arithmetic well-defined.
    let lhs =
        if predicate { point1 } else { EmbeddedCurvePoint::point_at_infinity() };
    let rhs =
        if predicate { point2 } else { EmbeddedCurvePoint::point_at_infinity() };

    assert(point_is_valid(lhs));
    assert(point_is_valid(rhs));

    let result = add_affine(lhs, rhs);
    if predicate {
        [result]
    } else {
        [EmbeddedCurvePoint::point_at_infinity()]
    }
}
