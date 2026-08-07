// Signed values crossing constrained function boundaries: as scalar params and
// returns, as array params and returns, and as the operand of a signed op that
// happens inside the callee rather than at the call site.
//
// The twin pair below is the deliberate part. `twin_signed` and `twin_unsigned`
// differ only in `i32` versus `u32`, so once the two collapse into one HLSSA
// type they become candidates for `merge_identical_functions`. That is one of
// the four predicted byte-level diffs for the stage that deletes `TypeExpr::U`,
// and this is the entry that should show it.

fn twin_signed(x: i32, y: i32) -> i32 {
    x + y
}

fn twin_unsigned(x: u32, y: u32) -> u32 {
    x + y
}

fn sdiv(x: i32, y: i32) -> i32 {
    x / y
}

fn srem(x: i32, y: i32) -> i32 {
    x % y
}

fn is_negative(x: i32) -> bool {
    x < 0
}

fn sum_signed(v: [i32; 4]) -> i32 {
    let mut s: i32 = 0;
    for i in 0..4 {
        s += v[i];
    }
    s
}

fn scale_signed(v: [i32; 3], k: i32) -> [i32; 3] {
    let mut out: [i32; 3] = [0; 3];
    for i in 0..3 {
        out[i] = v[i] * k;
    }
    out
}

fn main(a: i32, b: u32) {
    // a = -6, b = 6.
    assert_eq(twin_signed(a, 4), -2);
    assert_eq(twin_unsigned(b, 4), 10);

    // Signed division and remainder performed inside a callee.
    assert_eq(sdiv(a, 4), -1);
    assert_eq(srem(a, 4), -2);
    assert_eq(sdiv(a, -4), 1);

    // Signed comparison inside a callee, returning across the boundary.
    assert(is_negative(a));
    assert(!is_negative(-a));

    // Signed array as a parameter.
    let arr: [i32; 4] = [-1, 2, -3, 4];
    assert_eq(sum_signed(arr), 2);

    // Signed array as a return value, with a witness-derived scale factor.
    let scaled = scale_signed([1, -2, 3], a);
    assert_eq(scaled[0], -6);
    assert_eq(scaled[1], 12);
    assert_eq(scaled[2], -18);
}
