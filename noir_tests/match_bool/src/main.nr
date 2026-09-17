// True/False constructors, a bool arm with a wildcard, and arms that yield the same value
// (the elaborator compares arm bodies by `ExprId`, so `same` keeps two cases and no default).
fn pick(b: bool, x: Field, y: Field) -> Field {
    match b {
        true => x,
        false => y,
    }
}

fn only_true(b: bool) -> u8 {
    match b {
        true => 1,
        _ => 0,
    }
}

fn same(b: bool) -> u8 {
    match b {
        true => 7,
        false => 7,
    }
}

fn main(b: bool, x: Field, y: Field) {
    assert_eq(pick(b, x, y), x);
    assert_eq(pick(!b, x, y), y);
    assert_eq(only_true(b), 1);
    assert_eq(only_true(!b), 0);
    assert_eq(same(b), 7);
    assert_eq(same(!b), 7);
}
