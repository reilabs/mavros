fn main(x: Field, expected_sum: pub Field, expected_second: pub Field) {
    let mut pair = (x, 0);

    let first_ref = &mut pair.0;
    *first_ref = x + 1;

    let pair_ref = &mut pair;
    let second_ref = &mut pair_ref.1;
    *second_ref = x + 2;

    let second = pair_ref.1;

    assert_eq(second, expected_second);
    assert_eq(pair.0 + pair.1, expected_sum);
}
