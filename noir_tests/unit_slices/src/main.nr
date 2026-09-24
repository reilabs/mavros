// Unit-valued calls in slice operations must execute exactly once.
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
