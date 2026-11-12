use std::embedded_curve_ops::{EmbeddedCurvePoint, EmbeddedCurveScalar, multi_scalar_mul};

unconstrained fn main(scalar: u8, expected_x: u8, expected_y: [u8; 32]) {
    let scalar = scalar as Field;
    let expected_x = expected_x as Field;
    let expected_y = Field::from_le_bytes_checked(expected_y);

    let point = EmbeddedCurvePoint::generator();
    let scalar = EmbeddedCurveScalar { lo: scalar, hi: 0 };
    let result = multi_scalar_mul([point], [scalar]);

    assert_eq(result.x, expected_x);
    assert_eq(result.y, expected_y);
}
