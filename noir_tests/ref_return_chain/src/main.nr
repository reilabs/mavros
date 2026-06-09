// A returned ref forwarded up two call levels: the caller's witness write must
// back-taint both mid's and inner's view of the allocation.
fn inner() -> &mut Field {
    &mut 17
}

fn mid() -> &mut Field {
    inner()
}

fn main(x: Field) {
    let z = mid();
    *z = x;
    assert_eq(*z, x);
}
