// Tests u64 arithmetic and comparison operators.
// Needed by: date library (Unix timestamp arithmetic),
//            sha256 wrapping addition (u32 widened to u64).
fn main(timestamp: u64, seconds_per_day: u64) {
    // u64 addition
    let tomorrow = timestamp + seconds_per_day;
    assert_eq(tomorrow, 1758386939);

    // u64 subtraction
    let yesterday = timestamp - seconds_per_day;
    assert_eq(yesterday, 1758214139);

    // u64 multiplication
    let two_days: u64 = seconds_per_day * 2;
    assert_eq(two_days, 172800);

    // u64 division (needed for timestamp-to-date: days = timestamp / 86400)
    let total_days = timestamp / seconds_per_day;
    assert_eq(total_days, 20351);

    // u64 comparison (needed for date comparisons)
    assert(timestamp > seconds_per_day);
    assert(seconds_per_day < timestamp);
    assert(timestamp >= timestamp);
    assert(timestamp <= timestamp);
    assert(timestamp != seconds_per_day);
}
