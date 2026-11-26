fn main(
    x: u8,
    expected_sum: pub u8,
    expected_second: pub u8,
    expected_deep_inner: pub u8,
    expected_deep_total: pub u8,
    expected_mixed_tail: pub u8,
    expected_mixed_total: pub u8,
) {
    let x = x as Field;
    let expected_sum = expected_sum as Field;
    let expected_second = expected_second as Field;
    let expected_deep_inner = expected_deep_inner as Field;
    let expected_deep_total = expected_deep_total as Field;
    let expected_mixed_tail = expected_mixed_tail as Field;
    let expected_mixed_total = expected_mixed_total as Field;

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
