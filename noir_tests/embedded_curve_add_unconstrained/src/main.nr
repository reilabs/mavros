use std::embedded_curve_ops::{embedded_curve_add, EmbeddedCurvePoint};

unconstrained fn main(
    point_x: u8,
    point_y: [u8; 32],
    expected_x: [u8; 32],
    expected_y: [u8; 32],
) {
    let point_x = point_x as Field;
    let point_y = Field::from_le_bytes_checked(point_y);
    let expected_x = Field::from_le_bytes_checked(expected_x);
    let expected_y = Field::from_le_bytes_checked(expected_y);

    let point = EmbeddedCurvePoint { x: point_x, y: point_y };
    let result = embedded_curve_add(point, point);

    assert_eq(result.x, expected_x);
    assert_eq(result.y, expected_y);
}
