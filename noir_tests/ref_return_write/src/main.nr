// A callee-allocated ref returned to the caller: the witness written through `z` must
// back-taint make_ref's allocation (its pointee becomes WitnessOf), or reference
// invariance is broken.
fn make_ref() -> &mut u64 {
    &mut 17
}

fn main(x: u64, y: pub u64) {
    let z = make_ref();
    *z = x;
    assert(*z != y);
}
