// Tests multiple public inputs of different types.
// Needed by: passport main() has pub u64, pub u8, pub u8 parameters.
fn main(
    private_data: [u8; 4],
    current_date: pub u64,
    min_val: pub u8,
    max_val: pub u8,
) {
    // Sum the private bytes
    let mut sum: u64 = 0;
    for i in 0..4 {
        sum += private_data[i] as u64;
    }

    // Check sum is within public bounds
    assert(sum >= min_val as u64);
    assert(sum <= max_val as u64);

    // Check against public date (just verify it's nonzero)
    assert(current_date > 0);
}
