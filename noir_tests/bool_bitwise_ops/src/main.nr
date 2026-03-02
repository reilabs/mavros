// Tests boolean & and | operators on bool type.
// Needed by: passport compare_age uses (cond1) & (cond2) and (cond1) | (cond2)
//            in age range checking logic.
fn main(age: u8, min_age: u8, max_age: u8) {
    let above_min = age >= min_age;
    let below_max = age <= max_age;

    // Boolean AND
    let in_range = above_min & below_max;
    assert(in_range);

    // Boolean OR (at least one condition true)
    let is_adult = age >= 18;
    let is_senior = age >= 65;
    let has_discount = is_adult | is_senior;
    assert(has_discount);

    // Complex boolean expression
    let check_max = max_age != 0;
    let valid = above_min & (!check_max | below_max);
    assert(valid);
}
