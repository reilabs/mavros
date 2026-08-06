use std::embedded_curve_ops::{embedded_curve_add, EmbeddedCurvePoint};

fn main(
    point_x: Field,
    point_y: Field,
    expected_double_x: pub Field,
    expected_double_y: pub Field,
    expected_triple_x: pub Field,
    expected_triple_y: pub Field,
) {
    let point = EmbeddedCurvePoint { x: point_x, y: point_y };
    let infinity = EmbeddedCurvePoint::point_at_infinity();

    let doubled = embedded_curve_add(point, point);
    assert_eq(doubled.x, expected_double_x);
    assert_eq(doubled.y, expected_double_y);

    let tripled = embedded_curve_add(doubled, point);
    assert_eq(tripled.x, expected_triple_x);
    assert_eq(tripled.y, expected_triple_y);

    assert_eq(embedded_curve_add(point, infinity), point);
    assert_eq(embedded_curve_add(infinity, point), point);

    let inverse = EmbeddedCurvePoint { x: point.x, y: -point.y };
    assert(embedded_curve_add(point, inverse).is_infinite());

    // A false low-level predicate must suppress validation of inactive inputs.
    let invalid = EmbeddedCurvePoint { x: 1, y: 2 };
    let suppressed = std::mavros::replacements::embedded_curve_add::embedded_curve_add(
        invalid,
        invalid,
        false,
    )[0];
    assert(suppressed.is_infinite());
}
