// array SROA: a fixed-size array touched only at constant indices (ArraySet at
// index 0, ArrayGet at index 2, reads at 0 and 1) is a "Split" flow-group and
// should be peeled into one SSA value per cell, eliminating all array ops.
fn main(a: Field, b: Field, c: Field, expected: pub Field) {
    let mut arr = [a, b, c];
    arr[0] = arr[2] + 1;
    assert_eq(arr[0] + arr[1], expected);
}
