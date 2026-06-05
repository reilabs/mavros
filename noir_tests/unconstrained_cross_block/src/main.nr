unconstrained fn add_one(x: Field) -> Field {
    x + 1
}

fn main(x: Field) {
    let y = unsafe { add_one(x) };

    // Keep the use of y in a successor block.
    for _ in 0..1 {
        assert_eq(y, x + 1);
    }
}
