// Signed arithmetic under witness-dependent control flow, so the guarded
// signed paths in LowerPureGuards run: the signed overflow guard for
// add/sub/mul and the divmod guard's `lhs == INT_MIN && rhs == -1` case.
fn main(a: i32, b: i32, flag: bool, other: bool) {
    let mut acc: i32 = 0;
    if flag {
        acc = a + b;
        acc = acc - b;
        acc = acc * 2;
    } else {
        acc = a / b;
        acc = acc % 3;
    }
    assert_eq(acc, -10);

    // A guarded signed division whose divisor is itself witness-dependent.
    let mut q: i32 = 0;
    if other {
        q = a / b;
    } else {
        q = b / a;
    }
    assert_eq(q, -1);

    // Signed comparison feeding a witness-dependent branch.
    let mut sign: i32 = 0;
    if a < 0 {
        sign = -1;
    } else {
        sign = 1;
    }
    assert_eq(sign, -1);
}
