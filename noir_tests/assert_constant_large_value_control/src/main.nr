fn update_large_value(value: Field) -> Field {
    let mut values: [Field; 256] = [0; 256];
    values[255] = value;
    values[255]
}

fn main(x: Field) {
    assert(update_large_value(7) + x == x + 7);
}
