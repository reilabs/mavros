use std::embedded_curve_ops::{embedded_curve_add, EmbeddedCurvePoint};

fn main(
    point_x: u8,
    point_y: [u8; 32],
    expected_double_x: pub [u8; 32],
    expected_double_y: pub [u8; 32],
    expected_triple_x: pub [u8; 32],
    expected_triple_y: pub [u8; 32],
    random_lhs_x: [[u8; 32]; 4],
    random_lhs_y: [[u8; 32]; 4],
    random_rhs_x: [[u8; 32]; 4],
    random_rhs_y: [[u8; 32]; 4],
    random_sum_x: pub [[u8; 32]; 4],
    random_sum_y: pub [[u8; 32]; 4],
) {
    let point_x = point_x as Field;
    let point_y = Field::from_le_bytes_checked(point_y);
    let expected_double_x = Field::from_le_bytes_checked(expected_double_x);
    let expected_double_y = Field::from_le_bytes_checked(expected_double_y);
    let expected_triple_x = Field::from_le_bytes_checked(expected_triple_x);
    let expected_triple_y = Field::from_le_bytes_checked(expected_triple_y);
    let random_lhs_x = random_lhs_x.map(|b| Field::from_le_bytes_checked(b));
    let random_lhs_y = random_lhs_y.map(|b| Field::from_le_bytes_checked(b));
    let random_rhs_x = random_rhs_x.map(|b| Field::from_le_bytes_checked(b));
    let random_rhs_y = random_rhs_y.map(|b| Field::from_le_bytes_checked(b));
    let random_sum_x = random_sum_x.map(|b| Field::from_le_bytes_checked(b));
    let random_sum_y = random_sum_y.map(|b| Field::from_le_bytes_checked(b));

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

    // These points are deterministic random scalar multiples of the Grumpkin
    // generator. Their expected sums were computed independently.
    for i in 0..4 {
        let lhs = EmbeddedCurvePoint { x: random_lhs_x[i], y: random_lhs_y[i] };
        let rhs = EmbeddedCurvePoint { x: random_rhs_x[i], y: random_rhs_y[i] };
        let sum = embedded_curve_add(lhs, rhs);
        assert_eq(sum.x, random_sum_x[i]);
        assert_eq(sum.y, random_sum_y[i]);
    }
}
