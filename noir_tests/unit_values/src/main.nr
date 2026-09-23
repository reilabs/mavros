global UNIT: () = ();

struct Empty {}

fn tick(count: &mut Field) {
    *count = *count + 1;
}

fn identity(value: ()) {
    value
}

fn project(value: ((), Field)) {
    value.0
}

fn empty() -> Empty {
    Empty {}
}

fn check(index: u32) -> Field {
    let mut count: Field = 0;
    identity(tick(&mut count));
    let functions: [fn(()) -> (); 2] = [identity, |_| {}];
    functions[index](tick(&mut count));
    assert_eq(count, 2);

    let mut value = ();
    value = tick(&mut count);
    identity(value);
    identity(UNIT);
    identity(*&mut tick(&mut count));
    assert_eq(count, 4);

    let tuple = (tick(&mut count), { count = count + 1; }, 7);
    identity(tuple.0);
    project((tuple.1, tuple.2));
    assert_eq(tuple.2, 7);
    assert_eq(count, 6);

    let mut array = [tick(&mut count), tick(&mut count)];
    array[index] = tick(&mut count);
    identity(array[index]);
    let repeated = [tick(&mut count); 2];
    identity(repeated[index]);
    assert_eq(count, 10);

    let mut vector = @[(tick(&mut count), 1), ((), 2)];
    vector[index] = (tick(&mut count), 3);
    vector = vector.push_back((tick(&mut count), 4));
    vector = vector.push_front((tick(&mut count), 5));
    vector = vector.insert(index, (tick(&mut count), 6));
    assert_eq(vector.len(), 5);
    identity(vector[index].0);
    assert_eq(count, 15);

    identity(if index == 1 { tick(&mut count) } else { () });
    assert_eq(count, 16);
    assert_eq(tick(&mut count), ());
    assert(tick(&mut count) == ());
    assert(!(tick(&mut count) != ()));
    assert_eq(count, 19);

    // Both empty structs and unit have a materialized result before tuple elision.
    let _: Empty = empty();
    count
}

unconstrained fn check_unconstrained(index: u32) -> Field {
    check(index)
}

fn main(index: u32) {
    assert_eq(check(index), 19);
    // Safety: the result is checked against the expected count below.
    let count = unsafe { check_unconstrained(index) };
    assert_eq(count, 19);
}
