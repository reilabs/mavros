use std::field::bn254::{gt, lt};

unconstrained fn main(a: Field, b: pub Field) {
    // a = 3, b = 5
    assert(lt(a, b));
    assert(gt(b, a));
    assert(!lt(b, a));
    assert(!gt(a, b));
}
