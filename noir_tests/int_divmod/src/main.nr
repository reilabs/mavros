fn main(a: u32, b: u32) {
    let q = a / b;
    let r = a % b;
    // Verify division identity: (a/b)*b + (a%b) == a
    assert_eq(q * b + r, a);
    // Verify expected values for 17 / 5
    assert_eq(q, 3);
    assert_eq(r, 2);
    // Edge case: divisor = 1
    let q1 = a / 1;
    let r1 = a % 1;
    assert_eq(q1, a);
    assert_eq(r1, 0);
}
