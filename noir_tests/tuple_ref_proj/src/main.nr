fn main(
    x: Field,
    expected_sum: pub Field,
    expected_second: pub Field,
    expected_deep_inner: pub Field,
    expected_deep_total: pub Field,
    expected_mixed_tail: pub Field,
    expected_mixed_total: pub Field,
) {
    let mut pair = (x, 0);

    let first_ref = &mut pair.0;
    *first_ref = x + 1;

    let pair_ref = &mut pair;
    let second_ref = &mut pair_ref.1;
    *second_ref = x + 2;

    let second = pair_ref.1;

    assert_eq(second, expected_second);
    assert_eq(pair.0 + pair.1, expected_sum);

    let mut deep = ((x, (0, x + 3)), ((x + 4, 0), x + 5));

    let deep_ref = &mut deep;
    let deep_left_ref = &mut deep_ref.0.1.0;
    *deep_left_ref = x + 6;

    let deep_right_ref = &mut deep_ref.1.0.1;
    *deep_right_ref = deep_ref.0.1.0 + deep.1.1;

    deep.0.0 = deep_ref.1.0.1 + 1;

    let deep_inner = deep_ref.1.0.1;
    let deep_total =
        deep.0.0 + deep.0.1.0 + deep.0.1.1 + deep.1.0.0 + deep.1.0.1 + deep.1.1;

    assert_eq(deep_inner, expected_deep_inner);
    assert_eq(deep_total, expected_deep_total);

    let mut mixed = ((x, 0), (0, x + 1));
    let mixed_tail_ref = &mut mixed.1;

    (*mixed_tail_ref).0 = x + 7;
    mixed.0.1 = x + 8;
    (*mixed_tail_ref).1 = (*mixed_tail_ref).0 + mixed.0.1;

    let mixed_tail = (*mixed_tail_ref).1;
    let mixed_total = mixed.0.0 + mixed.0.1 + mixed.1.0 + mixed.1.1;

    assert_eq(mixed_tail, expected_mixed_tail);
    assert_eq(mixed_total, expected_mixed_total);
}
