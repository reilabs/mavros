fn main(a: i32, b: pub i32) {
    // Addition (positive)
    let sum: i32 = a + b;
    assert_eq(sum, 8);

    // Subtraction (positive result)
    let diff: i32 = b - a;
    assert_eq(diff, 2);

    // Subtraction (negative result)
    let neg_diff: i32 = a - b;
    assert_eq(neg_diff, -2);

    // Multiplication (positive * positive)
    let prod: i32 = a * b;
    assert_eq(prod, 15);

    // Multiplication producing negative result
    let neg_one: i32 = -1;
    let neg_a: i32 = neg_one * a;
    assert_eq(neg_a, -3);
    let neg_b: i32 = neg_one * b;
    assert_eq(neg_b, -5);

    // Addition of two negatives (requires carry)
    let neg_sum: i32 = neg_a + neg_b;
    assert_eq(neg_sum, -8);

    // Addition with negative literal
    let result: i32 = a + neg_one;
    assert_eq(result, 2);

    // Equality
    assert(a == a);
    assert(b == b);
    assert(a != b);
}
