use std::mem::{array_refcount, vector_refcount};

struct Empty {}

// The same helpers must select semantics from the caller's execution context.
fn array_count(a: [Field; 2]) -> u32 {
    array_refcount(a)
}

fn vector_count(a: [Field]) -> u32 {
    vector_refcount(a)
}

fn make_array(x: Field) -> [Field; 2] {
    assert_eq(x, 7);
    [x, x + 1]
}

fn make_vector(x: Field) -> [Field] {
    assert_eq(x, 7);
    [x, x + 1].as_vector()
}

unconstrained fn check_runtime(x: Field) -> u32 {
    let a = [x, x + 1];
    assert(array_count(a) > 0);
    assert(array_refcount(a) > 0);
    let v = a.as_vector();
    assert(vector_count(v) > 0);
    assert(vector_refcount(v) > 0);

    // Read temporaries at their last use, including empty allocations.
    assert_eq(array_refcount([x, x + 1]), 1);
    assert_eq(vector_refcount([x, x + 1].as_vector()), 1);
    let empty_array: [Field; 0] = [];
    let empty_vector: [Field] = [];
    assert_eq(array_refcount(empty_array), 1);
    assert_eq(vector_refcount(empty_vector), 1);
    // Generic builtins also accept composite and zero-sized elements.
    assert_eq(array_refcount([(x, x + 1)]), 1);
    assert_eq(vector_refcount([(x, x + 1)].as_vector()), 1);
    assert_eq(array_refcount([[x, x + 1]]), 1);
    assert_eq(vector_refcount([[x, x + 1]].as_vector()), 1);
    // Mavros erases arrays of unit, so these have no backing allocation.
    assert_eq(array_refcount([Empty {}; 2]), 0);
    1
}

fn main(x: Field) {
    assert_eq(array_count([x, x + 1]), 0);
    assert_eq(vector_count([x, x + 1].as_vector()), 0);
    assert_eq(array_refcount(make_array(x)), 0);
    assert_eq(vector_refcount(make_vector(x)), 0);
    // Safety: the hint returns a fixed value checked below.
    let result = unsafe { check_runtime(x) };
    assert_eq(result, 1);
}
