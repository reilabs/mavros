// Runtime bounds exercise the endpoint check in both the body fallthrough and continue paths.
unconstrained fn count_range(lo: u8, hi: u8, skip_endpoint: bool) -> u32 {
    let mut count = 0;
    for i in lo..=hi {
        count += 1;
        if skip_endpoint & (i == hi) {
            continue;
        }
    }
    count
}

unconstrained fn signed_continue() -> u32 {
    let mut count = 0;
    for i in 125i8..=127i8 {
        count += 1;
        if i != 126 {
            continue;
        }
    }
    count
}

unconstrained fn nested_break_continue() -> u32 {
    let mut count = 0;
    for _ in 254u8..=255u8 {
        for j in 254u8..=255u8 {
            count += 1;
            if j == 255 {
                break;
            }
            continue;
        }
        continue;
    }
    count
}

fn main(lo: u8, hi: u8) {
    // The complete u8 range includes 256 values, with no end + 1 or final increment.
    let mut count: u32 = 0;
    let mut sum: u32 = 0;
    for i in 0u8..=255u8 {
        count += 1;
        sum += i as u32;
    }
    assert(count == 256);
    assert(sum == 32640);

    let mut signed_sum: i32 = 0;
    for i in 126i8..=127i8 {
        signed_sum += i as i32;
    }
    assert(signed_sum == 253);

    // The same lowering applies to wider integer types and singleton ranges.
    let mut wide_count: u32 = 0;
    for _ in 18446744073709551614u64..=18446744073709551615u64 {
        wide_count += 1;
    }
    for _ in 9223372036854775807i64..=9223372036854775807i64 {
        wide_count += 1;
    }
    assert(wide_count == 3);

    let mut empty_count: u32 = 0;
    for _ in 255u8..=254u8 {
        empty_count += 1;
    }
    for _ in 0i8..=-1i8 {
        empty_count += 1;
    }
    assert(empty_count == 0);

    assert(unsafe { count_range(lo, hi, false) } == 3);
    assert(unsafe { count_range(lo, hi, true) } == 3);
    assert(unsafe { count_range(hi, hi, true) } == 1);
    assert(unsafe { count_range(hi, lo, true) } == 0);
    assert(unsafe { signed_continue() } == 3);
    assert(unsafe { nested_break_continue() } == 4);
}
