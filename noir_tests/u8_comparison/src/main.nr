// Tests comparison operators (<, >, <=, >=, !=) on u8 witness values.
// Needed by: compare_age checks min_age <= age <= max_age,
//            check_expiry checks current_date < expiry_date,
//            check_zero_padding verifies bytes are zero.
fn main(age: u8, min_age: pub u8, max_age: pub u8) {
    assert(age >= min_age);
    assert(age <= max_age);
    assert(age != min_age);
    assert(min_age < max_age);
}
