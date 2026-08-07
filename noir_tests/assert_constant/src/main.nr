use std::assert_constant;

global LARGE_CONSTANT: [Field; 5000] = [7; 5000];

struct Pair {
    left: Field,
    right: [Field; 2],
}

fn requires_constant(value: Field) {
    assert_constant(value);
}

fn forwards_constant(value: Field) {
    requires_constant(value);
}

fn plus_one(value: Field) -> Field {
    value + 1
}

fn dynamic_vector(value: Field) -> [Field] {
    [value; 4].as_vector()
}

fn main(x: Field) {
    forwards_constant(11);
    forwards_constant(12);
    assert_constant(plus_one(20));
    assert_constant(dynamic_vector(x).len());

    assert_constant(LARGE_CONSTANT);
    assert_constant((1, 2));
    assert_constant(Pair { left: 3, right: [4, 5] });

    let dynamic_array = [x; 17];
    let dynamic_vector = dynamic_array.as_vector();
    assert_constant(dynamic_array.len());
    assert_constant(dynamic_vector.len());
}
