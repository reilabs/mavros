// `for` loops whose index is a signed integer.
//
// The frontend desugars a `for` into three operations on the index -- the exit
// test in the header, the increment it falls off the end of the body into, and
// the increment `continue` jumps through -- plus a fourth, the bump of the end
// bound that makes an inclusive range exclusive. All four have to be built at
// the index's own signedness, and the constant `1` they step by has to be
// tagged the same way the index is: `Types` gives a binary operation the type
// `lhs.get_arithmetic_result_type(rhs)`, which has no mixed `U`/`I` case and
// panics outright. A `u64` step against an `i64` index therefore did not
// miscompile, it made the whole function untypeable, and every `for` over a
// signed range died with `Cannot perform arithmetic on types i64 and u64`
// before reaching any backend.
//
// Each group below reaches one of those sites, and the assertions pin the
// arithmetic against Noir's own execution via the `Correct` lanes.

// `continue` is the second increment site, and Noir allows it only in
// unconstrained code.
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

    // Inclusive range, which bumps the end bound by one before the header is
    // built -- a fourth operation on a signed value, at a narrower width.
    let mut n: i8 = 0;
    for j in -2i8..=2i8 {
        n += j + y;
    }
    assert(n == 5);

    // The `continue` increment, on a range that straddles zero.
    let c = unsafe { skip_zero(-4, 4) };
    assert(c == -4);
}
