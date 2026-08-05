use std::assert_constant;

unconstrained fn requires_constant(value: Field) {
    // Mavros currently accepts this even though `value` is witness-derived: WTI deliberately omits
    // unconstrained-only callees. This test pins that documented divergence from Noir semantics.
    assert_constant(value);
}

fn main(x: Field) {
    unsafe { requires_constant(x) };
}
