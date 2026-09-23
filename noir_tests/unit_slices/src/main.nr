// Requires zero-sized slice lowering before tuple elision (PR #365).
// This exercises unit materialization together with length and bounds preservation.
fn tick(count: &mut Field) {
    *count = *count + 1;
}

fn check(index: u32) -> Field {
    let mut count: Field = 0;
    let mut values = @[(), tick(&mut count)];
    values[index] = tick(&mut count);
    values = values.push_back(tick(&mut count));
    values = values.push_front(());
    // Insertion at the end is valid; get/set/remove use a strict upper bound.
    values = values.insert(values.len(), tick(&mut count));
    assert_eq(values.len(), 5);
    let repeated: [(); 2] = [tick(&mut count); 2];
    assert_eq(repeated[index], values[index]);
    let (without_last, last) = values.pop_back();
    assert_eq(last, ());
    let (first, without_first) = without_last.pop_front();
    assert_eq(first, ());
    let (remaining, removed) = without_first.remove(index);
    assert_eq(removed, ());
    assert_eq(remaining.len(), 2);
    let from_array = repeated.as_vector();
    assert_eq(from_array.len(), 2);
    let empty: [()] = @[];
    assert_eq(empty.len(), 0);
    assert_eq(empty.push_back(()).len(), 1);
    assert_eq(count, 5);
    count
}

unconstrained fn check_unconstrained(index: u32) -> Field {
    check(index)
}

fn main(index: u32) {
    assert_eq(check(index), 5);
    // Safety: the result is checked against the expected evaluation count.
    let count = unsafe { check_unconstrained(index) };
    assert_eq(count, 5);
}
