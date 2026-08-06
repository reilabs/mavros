// Signed division and remainder at every supported width, with both witness
// and constant operands. Noir truncates the quotient toward zero, so the
// remainder takes the dividend's sign.
//
// `INT_MIN / -1` is the one combination that genuinely overflows and is never
// evaluated here; `noir_tests/` is discovered as execution-success only, so a
// deliberately-failing case has no home in this corpus.
fn main(a: i32, b: i32, c: i64, m: i64, d: i8, g: i8, h: i16, j: i16) {
    // a = -17, b = 5.
    let q = a / b;
    let r = a % b;
    assert_eq(q * b + r, a);
    assert_eq(q, -3);
    assert_eq(r, -2);

    // Negative divisor.
    let nb = -b;
    assert_eq(a / nb, 3);
    assert_eq(a % nb, -2);

    // Positive dividend, negative divisor.
    let na = -a;
    assert_eq(na / nb, -3);
    assert_eq(na % nb, 2);

    // Divisors of 1 and -1. Safe here because `a` is not i32::MIN.
    assert_eq(a / 1, a);
    assert_eq(a % 1, 0);
    assert_eq(a / (-1), na);

    // i64, witness divisor. c = -100, m = 7.
    let q64 = c / m;
    let r64 = c % m;
    assert_eq(q64 * m + r64, c);
    assert_eq(q64, -14);
    assert_eq(r64, -2);

    // i64, constant divisor: the constant-folding shape rather than the
    // witness lowering.
    assert_eq(c / 7, -14);
    assert_eq(c % 7, -2);

    // i8, witness divisor. d = -100, g = 3.
    assert_eq(d / g, -33);
    assert_eq(d % g, -1);

    // i16, witness divisor. h = -1000, j = 7.
    assert_eq(h / j, -142);
    assert_eq(h % j, -6);

    // The same narrow widths with constant operands, so the signed
    // constant-folding path runs as well as the witness lowering.
    let dc: i8 = -100;
    assert_eq(dc / 3, -33);
    assert_eq(dc % 3, -1);
    let hc: i16 = -1000;
    assert_eq(hc / 7, -142);
    assert_eq(hc % 7, -6);
}
