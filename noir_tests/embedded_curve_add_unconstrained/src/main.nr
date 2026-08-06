use std::embedded_curve_ops::{embedded_curve_add, EmbeddedCurvePoint};

unconstrained fn main(
    point_x: Field,
    point_y: Field,
    expected_x: Field,
    expected_y: Field,
) {
    let point = EmbeddedCurvePoint { x: point_x, y: point_y };
    let result = embedded_curve_add(point, point);

    assert_eq(result.x, expected_x);
    assert_eq(result.y, expected_y);
}
