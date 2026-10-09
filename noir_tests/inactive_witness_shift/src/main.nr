// Guarded witness shifts, taken with the guard inactive.
//
// A shift amount at or past the width is a runtime error, so the witness lowering checks it, and
// that check must be guarded: `a >> n`, with `n = 40` on a u32, sits in an inactive branch and must
// not fail the program.
//
// The left shifts overflow, which Noir allows (it wraps), so they must stay satisfiable whatever
// the guard and give the truncated value when it is live. A lowering that range-checks the product
// instead of truncating fails both ways.
//
// Upstream shift tests all shift by a constant, which folds before this lowering runs.
fn main(active: bool, live: bool, e: i8, k: i8, k2: i8, a: u32, n: u32) {
    // `e` is 64, so `e << 1` overflows i8. Inactive, so it must not be reported.
    let overflowing_shl = if active { e << 1 } else { 0 };
    assert_eq(overflowing_shl, 0);

    // The same overflow reached through a witness amount, which is the case that
    // cannot fold: `k` is 7.
    let overflowing_witness_shl = if active { e << k } else { 0 };
    assert_eq(overflowing_witness_shl, 0);

    // An amount at or above the width is a runtime error in Noir, so the check
    // that rejects it has to be guarded as well. `n` is 40, against a u32.
    let overshift = if active { a >> n } else { 0 };
    assert_eq(overshift, 0);

    // The live counterparts, so the guarded path is exercised computing an
    // answer rather than only proven vacuous.
    let shifted = if live { a >> 2 } else { 0 };
    assert_eq(shifted, 10);
    let signed_shifted = if live { e >> k2 } else { 0 };
    assert_eq(signed_shifted, 8);

    // And a live shift that overflows, so the wrapping answer is checked under
    // a guard rather than only proven harmless. 64:i8 << 1 is -128.
    let live_wrapping_shl = if live { e << 1 } else { 0 };
    assert_eq(live_wrapping_shl, -128);
}
