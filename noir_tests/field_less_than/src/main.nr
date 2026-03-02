// Tests the field_less_than builtin intrinsic.
// The passport program uses this (1 call site) for field element comparison.
use std::field::bn254::lt as field_lt;

fn main(a: Field, b: Field) {
    // a = 5, b = 10, so a < b should be true
    assert(field_lt(a, b));
    // b < a should be false
    assert(!field_lt(b, a));
}
