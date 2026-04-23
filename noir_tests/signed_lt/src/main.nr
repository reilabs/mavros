fn main(a: i32, b: i32, c: i32) {
    // Different signs: -7 < 3
    assert(c < a);

    // Same sign positive: 3 < 5
    assert(a < b);

    // Same sign negative: -7 < -1
    // We encode -1 via (c + 6) since c = -7
    let minus_one = c + 6; // -7 + 6 = -1, but compiler may fold
    // Instead, use the inputs directly
    // a=3, b=5, c=-7
    // -7 < -1 is true; -1 = c + a + b - 1 = -7 + 3 + 5 - 1 = 0... no
    // Just assert things we can derive from a=3, b=5, c=-7

    // Negative < Negative: c < c + a  i.e. -7 < -4
    assert(c < c + a);

    // Not less than (equal): a == a
    assert(a < a == false);

    // Not less than (greater): b > a
    assert(b < a == false);

    // Positive < Negative is false
    assert(a < c == false);
}
