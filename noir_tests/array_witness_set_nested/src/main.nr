// Witness-index set of a nested element into a pure array: exercises the build_array_loop
// branch of gen_witness_array_set with an array-typed element.
fn main(w: u32, e00: pub u8, e10: pub u8) {
    let e00 = e00 as Field;
    let e10 = e10 as Field;

    let mut a: [[Field; 2]; 2] = [[1, 2], [3, 4]];
    a[w] = [5, 6];
    assert_eq(a[0][0], e00);
    assert_eq(a[1][0], e10);
    assert_eq(a[w][1], 6);
}
