// True/False constructors, a bool arm with a wildcard, and identical arms that the
// elaborator merges into a default with no cases left.
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
