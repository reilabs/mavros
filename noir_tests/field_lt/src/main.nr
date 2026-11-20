use std::field::bn254::{gt, lt};

unconstrained fn main(a: u8, b: pub u8) {
    let a = a as Field;
    let b = b as Field;

    // a = 3, b = 5
    assert(lt(a, b));
    assert(gt(b, a));
    assert(!lt(b, a));
    assert(!gt(a, b));
}
