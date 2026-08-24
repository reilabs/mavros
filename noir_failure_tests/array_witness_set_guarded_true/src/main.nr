// The guard-on counterpart of `noir_tests/array_witness_set_guarded_false`, so that test cannot
// pass vacuously by the check having been dropped altogether: same program, guard true, and the
// out-of-range write must still be rejected.
fn main(w: Field, i: u32, expected0: pub Field) {
    let mut a: [Field; 3] = [1, 2, 3];
    if w == 1 {
        a[i] = 9;
    }
    assert_eq(a[0], expected0);
}
