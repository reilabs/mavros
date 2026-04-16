use date::Date;

global DG1_TO_MRZ_OFFSET: u32 = 5;
global ID_CARD_MRZ_BIRTHDATE_INDEX: u32 = 30;
global PASSPORT_MRZ_BIRTHDATE_INDEX: u32 = 57;

fn main(
    dg1: [u8; 95],
    current_date: pub u64,
    min_age_required: pub u8,
    max_age_required: pub u8,
) {
    assert((max_age_required < 100) & (min_age_required < 100), "Age must be less than 100");

    let current_date = Date::from_timestamp(current_date);
    let mut mrz = [0 as u8; 90];
    for i in 0..90 {
        mrz[i] = dg1[i + DG1_TO_MRZ_OFFSET];
    }

    let mut birthdate_bytes = [0 as u8; 6];
    let birthdate_index = if (dg1[93] != 0) & (dg1[94] != 0) {
        ID_CARD_MRZ_BIRTHDATE_INDEX
    } else {
        PASSPORT_MRZ_BIRTHDATE_INDEX
    };
    for i in 0..6 {
        birthdate_bytes[i] = mrz[birthdate_index + i];
    }
    let birthdate = Date::from_bytes_short_year(birthdate_bytes, current_date);

    assert(
        (min_age_required != 0) | (max_age_required != 0),
        "Either min or max age must be non-zero",
    );

    if (min_age_required != 0) & (max_age_required == 0) {
        assert(
            current_date.gte(birthdate.add_years(min_age_required as u32)),
            "Age is not above or equal to min age",
        );
    } else if (max_age_required != 0) & (min_age_required == 0) {
        assert(
            current_date.lt(birthdate.add_years(max_age_required as u32)),
            "Age is not below max age",
        );
    } else {
        assert(min_age_required <= max_age_required, "Min age must be less than or equal to max age");
        assert(
            current_date.gte(birthdate.add_years(min_age_required as u32)),
            "Age is not above or equal to min age",
        );
        assert(
            current_date.lt(birthdate.add_years((max_age_required + 1) as u32)),
            "Age is not below max age",
        );
    }
}
