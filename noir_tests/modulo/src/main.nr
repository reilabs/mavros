// Tests the modulo (%) operator on u32 and u64.
// Needed by: date library (timestamp to date conversion),
//            SHA-256 wrapping addition pattern.
fn main(a: u32, b: u32, c: u64, d: u64) {
    // u32 modulo
    let r1 = a % b;
    assert_eq(r1, 3); // 17 % 7 = 3

    // u64 modulo (timestamp-like: seconds in a day)
    let r2 = c % d;
    assert_eq(r2, 2); // 100 % 7 = 2
}
