use std::embedded_curve_ops::{EmbeddedCurvePoint, EmbeddedCurveScalar, multi_scalar_mul};

unconstrained fn main(scalar: Field, expected_x: Field, expected_y: Field) {
    let point = EmbeddedCurvePoint::generator();
    let scalar = EmbeddedCurveScalar { lo: scalar, hi: 0 };
    let result = multi_scalar_mul([point], [scalar]);

    assert_eq(result.x, expected_x);
    assert_eq(result.y, expected_y);
}
