fn increment(x: Field) -> Field {
    x + 1
}

fn double(x: Field) -> Field {
    x * 2
}

fn check(index: u32, x: Field) {
    let functions = [increment, double].as_vector();
    assert_eq(functions[index - 1](x), x + 1);
    assert_eq(functions[index](x), x * 2);

    let lambdas = [|a: Field| a + 10, |a: Field| a * 17].as_vector();
    assert_eq(lambdas[index - 1](x), x + 10);
    assert_eq(lambdas[index](x), x * 17);

    let repeated = [|a: Field| a + 7; 2].as_vector();
    assert_eq(repeated[index](x), x + 7);

    let units: [fn(()) -> ()] = [|_: ()| {}].as_vector();
    units[0](());
    units[index - 1](());
}

fn main(index: u32, x: Field) {
    check(index, x);
    // Safety: the same checks are enforced by the constrained call above.
    unsafe { check(index, x) };
}
