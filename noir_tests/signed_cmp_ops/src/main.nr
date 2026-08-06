// `<=`, `>` and `>=` all lower to `Lt` plus a `Not`, so each exercises the
// signed comparison path from a different direction.
fn main(a: i32, b: i32, c: i8, d: i8) {
    // a = -5, b = 3: opposite sides of the sign boundary.
    assert(a < b);
    assert(a <= b);
    assert(b > a);
    assert(b >= a);
    assert(!(b < a));
    assert(!(b <= a));

    // Reflexive cases.
    assert(a <= a);
    assert(a >= a);
    assert(!(a < a));

    // i8 extremes: MIN is the value whose unsigned reading is largest.
    assert(c < d);
    assert(c <= d);
    assert(!(d < c));
    assert(d > c);
    assert(c <= c);
    assert(d >= d);

    // Against zero.
    assert(a < 0);
    assert(!(b < 0));
    assert(c < 0);
    assert(d > 0);

    // Both negative.
    let a2 = a - 1;
    assert(a2 < a);
    assert(!(a < a2));
}
