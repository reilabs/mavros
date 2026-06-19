// array SROA guard: a witness-dependent (dynamic) index collapses the array
// flow-group to a single AllElems cell, so no peeling happens. The result must
// still be correct; this guards the collapse path.
fn main(a: Field, b: Field, c: Field, d: Field, i: u32, expected: pub Field) {
    let arr = [a, b, c, d];
    assert_eq(arr[i], expected);
}
