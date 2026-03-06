// Test signed integer literals and arithmetic (i32, i64).
// The passport program uses signed types for date arithmetic:
//   - duration_days: i64 computed from year/month/day differences
//   - comparisons: duration >= 0, duration < threshold
fn main(year: u32, day: u32) {
    // i32 signed literal
    let a: i32 = -1;
    let b: i32 = year as i32;
    assert(b > a);

    // i32 arithmetic
    let c: i32 = 10;
    let d: i32 = -5;
    assert(c + d == 5);

    // i64 signed arithmetic (date duration pattern)
    let year_i64 = year as i64;
    let day_i64 = day as i64;
    let duration: i64 = year_i64 * 365 + day_i64;
    assert(duration > 0);

    // i64 subtraction producing negative result
    let past: i64 = 100;
    let future: i64 = 200;
    let diff = past - future;
    assert(diff < 0);
}
