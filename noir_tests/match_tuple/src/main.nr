// A tuple pattern is a single untested case that emits no condition; nested literals and
// bools become switches on the bound elements.
fn swap(t: (u32, u32)) -> (u32, u32) {
    match t {
        (a, b) => (b, a),
    }
}

fn first_zero(t: (u32, u32)) -> u32 {
    match t {
        (0, b) => b,
        (a, _) => a,
    }
}

fn both(a: bool, b: bool) -> u8 {
    match (a, b) {
        (true, true) => 3,
        (true, false) => 2,
        (false, true) => 1,
        (false, false) => 0,
    }
}

fn main(x: u32, y: u32, b: bool) {
    let (p, q) = swap((x, y));
    assert_eq(p, y);
    assert_eq(q, x);
    assert_eq(first_zero((0, y)), y);
    assert_eq(first_zero((x, y)), x);
    assert_eq(both(b, b), 3);
    assert_eq(both(b, !b), 2);
    assert_eq(both(!b, b), 1);
    assert_eq(both(!b, !b), 0);
}
