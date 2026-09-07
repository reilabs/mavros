use std::hint::black_box;

struct Payload {
    scalar: Field,
    array: [Field; 2],
}

fn set_ref(c: &mut Field, d: &mut Field) {
    *c = *d;
}

// Pin value semantics, including argument evaluation and temporary references.
// The optimization hint is best-effort; no particular optimized shape is required.
fn check_shapes(x: Field) -> Field {
    black_box(());
    let mut evaluations: Field = 0;
    black_box({
        evaluations = evaluations + 1;
    });
    assert_eq(evaluations, 1);

    let scalar = black_box({
        evaluations = evaluations + 1;
        x
    });
    assert_eq(evaluations, 2);
    assert_eq(scalar, x);

    let array = black_box([x, x + 1]);
    assert_eq(array[0], x);
    assert_eq(array[1], x + 1);

    let payload = black_box(Payload { scalar: x, array });
    assert_eq(payload.scalar, x);
    assert_eq(payload.array[0], x);
    assert_eq(payload.array[1], x + 1);

    let tuple = black_box((x, array));
    assert_eq(tuple.0, x);
    assert_eq(tuple.1[0], x);
    assert_eq(tuple.1[1], x + 1);

    let slice = black_box(array.as_vector());
    assert_eq(slice.len(), 2);
    assert_eq(slice[0], x);
    assert_eq(slice[1], x + 1);

    let mut c: Field = 0;
    let mut d = x;
    set_ref(&mut black_box(c), &mut d);
    assert_eq(c, 0);
    assert_eq(d, x);
    // A reference to the actual local must still update it.
    set_ref(&mut c, &mut d);
    assert_eq(c, x);
    c
}

unconstrained fn check_unconstrained(x: Field) -> Field {
    check_shapes(x)
}

fn main(x: Field) {
    assert_eq(check_shapes(x), x);
    // Safety: the returned identity value is checked against the input below.
    let result = unsafe { check_unconstrained(x) };
    assert_eq(result, x);
}
