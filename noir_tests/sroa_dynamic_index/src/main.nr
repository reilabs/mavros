// array SROA guard: a witness-dependent (dynamic) index collapses the array
// flow-group to a single AllElems cell, so no peeling happens. The result must
// still be correct; this guards the collapse path.
fn main(a: u8, b: u8, c: u8, d: u8, i: u32, expected: pub u8) {
    let a = a as Field;
    let b = b as Field;
    let c = c as Field;
    let d = d as Field;
    let expected = expected as Field;

    let arr = [a, b, c, d];
    assert_eq(arr[i], expected);
}
