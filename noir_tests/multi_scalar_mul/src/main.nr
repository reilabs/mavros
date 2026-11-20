use std::embedded_curve_ops::{EmbeddedCurvePoint, EmbeddedCurveScalar, multi_scalar_mul};

fn main(
    scalar: u8,
    expected_x: pub u8,
    expected_y: pub [u8; 32],
    expected_double_x: pub [u8; 32],
    expected_double_y: pub [u8; 32],
    expected_high_limb_x: pub [u8; 32],
    expected_high_limb_y: pub [u8; 32],
) {
    let scalar = scalar as Field;
    let expected_x = expected_x as Field;
    let expected_y = Field::from_le_bytes_checked(expected_y);
    let expected_double_x = Field::from_le_bytes_checked(expected_double_x);
    let expected_double_y = Field::from_le_bytes_checked(expected_double_y);
    let expected_high_limb_x = Field::from_le_bytes_checked(expected_high_limb_x);
    let expected_high_limb_y = Field::from_le_bytes_checked(expected_high_limb_y);

    let point = EmbeddedCurvePoint::generator();
    let scalar = EmbeddedCurveScalar { lo: scalar, hi: 0 };
    let result = multi_scalar_mul([point], [scalar]);

    assert_eq(result.x, expected_x);
    assert_eq(result.y, expected_y);

    let doubled = multi_scalar_mul([point, point], [scalar, scalar]);
    assert_eq(doubled.x, expected_double_x);
    assert_eq(doubled.y, expected_double_y);

    // Exercise the upper 128-bit scalar limb rather than only BN254 Field
    // decompositions, whose high limbs are usually small.
    let high_limb_scalar = EmbeddedCurveScalar { lo: 0, hi: 23 };
    let high_limb_result = multi_scalar_mul([point], [high_limb_scalar]);
    assert_eq(high_limb_result.x, expected_high_limb_x);
    assert_eq(high_limb_result.y, expected_high_limb_y);

    // The low-level contract returns infinity and suppresses input validation
    // when its predicate is false.
    let invalid_point = EmbeddedCurvePoint { x: 1, y: 2 };
    let invalid_scalar = EmbeddedCurveScalar { lo: -1, hi: -1 };
    let suppressed = std::mavros::replacements::multi_scalar_mul::multi_scalar_mul(
        [invalid_point],
        [invalid_scalar],
        false,
    )[0];
    assert(suppressed.is_infinite());
}
