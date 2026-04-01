fn main(a: i32, b: pub i32) {
    // Addition
    let sum: i32 = a + b;
    assert_eq(sum, 8);

    // Subtraction (positive result)
    let diff: i32 = b - a;
    assert_eq(diff, 2);

    // Subtraction (negative result)
    let neg_diff: i32 = a - b;
    assert_eq(neg_diff, -2);

    // Multiplication
    let prod: i32 = a * b;
    assert_eq(prod, 15);

    // Equality
    assert(a == a);
    assert(b == b);
    assert(a != b);

    // Negative literal
    let neg_one: i32 = -1;
    let result: i32 = a + neg_one;
    assert_eq(result, 2);
}
