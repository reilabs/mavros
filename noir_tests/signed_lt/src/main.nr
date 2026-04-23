fn main(a: i64, b: i64, c: i64) {
    // Different signs: -7 < 3
    assert(c < a);

    // Same sign positive: 3 < 5
    assert(a < b);

    // Negative < Negative: c < c + a  i.e. -7 < -4
    assert(c < c + a);

    // Not less than (equal): a == a
    assert(a < a == false);

    // Not less than (greater): b > a
    assert(b < a == false);

    // Positive < Negative is false
    assert(a < c == false);
}
