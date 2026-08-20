// Unit-returning main: no guard slot, the blob is exactly [x].
fn main(x: Field) -> pub () {
    assert(x != 0);
}
