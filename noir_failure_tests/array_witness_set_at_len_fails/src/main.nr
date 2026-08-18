// `i == len` is the first out-of-range index, and the one an off-by-one in the bounds check
// would let through. Same program as `noir_tests/array_witness_set_boundary`, one index higher.
fn main(i: u32, expected2: pub Field) {
    let mut a: [Field; 3] = [1, 2, 3];
    a[i] = 9;
    assert_eq(a[2], expected2);
}
