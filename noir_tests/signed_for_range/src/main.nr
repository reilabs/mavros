// `for` loops whose index is a signed integer.
//
// Lowering preserves the index's signedness in the header comparison and the
// shared increment reached by both body fallthrough and `continue`. For an
// inclusive range with a constant endpoint below the maximum, it also uses
// signed arithmetic to compute the safe exclusive bound. The step constant
// must have the index's width.
//
// These cases exercise exclusive and inclusive signed header comparisons,
// including the constant-endpoint fast path, and both routes to the increment.
// The assertions check the arithmetic through the runner's `Correct` lanes.

// `continue` reaches the shared increment; Noir allows it only in unconstrained code.
unconstrained fn skip_zero(lo: i64, hi: i64) -> i64 {
    let mut c: i64 = 0;
    for k in lo..hi {
        if k == 0 {
            continue;
        }
        c += k;
    }
    c
}

fn main(x: i64, y: i8) {
    // Exclusive range starting below zero, so the index really does hold
    // negative values rather than merely being declared able to. The exit test
    // is the case that matters here: read unsigned, `-3i64` is a little under
    // 2^64 and the loop would not run at all.
    let mut acc: i64 = 0;
    for i in -3i64..3i64 {
        acc += i + x;
    }
    assert(acc == 15);

    // The constant endpoint becomes 3i8, used by the header's signed `<` comparison.
    let mut n: i8 = 0;
    for j in -2i8..=2i8 {
        n += j + y;
    }
    assert(n == 5);

    // The `continue` increment, on a range that straddles zero.
    let c = unsafe { skip_zero(-4, 4) };
    assert(c == -4);
}
