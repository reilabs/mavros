// The last in-range slot must still be writable. Pairs with
// `noir_failure_tests/array_witness_set_at_len_fails`, which is the same program one index
// higher: together they pin the bound at exactly `len` rather than one either side of it. That
// off-by-one is easy to get wrong in the array arm, whose check is `sum(hit) == 1` over the
// visited slots rather than an ordering comparison against the length.
fn main(i: u32, expected2: pub Field) {
    let mut a: [Field; 3] = [1, 2, 3];
    a[i] = 9;
    assert_eq(a[2], expected2);
}
