use std::embedded_curve_ops::EmbeddedCurvePoint;
use std::hash::derive_generators;

fn derive_after_constant_arithmetic(offset: u32) -> [EmbeddedCurvePoint; 1] {
    derive_generators("DEFAULT_DOMAIN_SEPARATOR".as_bytes(), offset + 1)
}

fn derive_point(offset: u32) -> EmbeddedCurvePoint {
    derive_after_constant_arithmetic(offset)[0]
}

fn apply_derivation(
    derive: fn(u32) -> EmbeddedCurvePoint,
    offset: u32,
) -> EmbeddedCurvePoint {
    derive(offset)
}

fn main() {
    let generator = derive_after_constant_arithmetic(0)[0];
    assert_eq(
        generator.x,
        0x054aa86a73cb8a34525e5bbed6e43ba1198e860f5f3950268f71df4591bde402,
    );
    assert_eq(
        generator.y,
        0x209dcfbf2cfb57f9f6046f44d71ac6faf87254afc7407c04eb621a6287cac126,
    );

    // The same wrapper must specialize independently at a second constant call site.
    let next_generator = derive_after_constant_arithmetic(1)[0];
    assert_eq(
        next_generator.x,
        0x1c44f2a5207c81c28a8321a5815ce8b1311024bbed131819bbdaf5a2ada84748,
    );
    assert_eq(
        next_generator.y,
        0x03aaee36e6422a1d0191632ac6599ae9eba5ac2c17a8c920aa3caf8b89c5f8a8,
    );

    // A targeted wrapper retains its function identity and specializes through a higher-order
    // helper once the concrete function pointer and argument reach the call site.
    let indirect_generator = apply_derivation(derive_point, 0);
    assert_eq(indirect_generator.x, generator.x);
    assert_eq(indirect_generator.y, generator.y);
}
