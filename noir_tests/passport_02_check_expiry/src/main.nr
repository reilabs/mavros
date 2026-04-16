use date::Date;

global DG1_TO_MRZ_OFFSET: u32 = 5;
global ID_CARD_MRZ_EXPIRY_DATE_INDEX: u32 = 38;
global PASSPORT_MRZ_EXPIRY_DATE_INDEX: u32 = 65;

fn main(
    dg1: [u8; 95],
    current_date: pub u64,
) {
    let current_date = Date::from_timestamp(current_date);
    let threshold_year = current_date.add_years(30);
    let mut mrz = [0 as u8; 90];
    for i in 0..90 {
        mrz[i] = dg1[i + DG1_TO_MRZ_OFFSET];
    }

    let expiry_date = if (dg1[93] != 0) & (dg1[94] != 0) {
        let mut expiry_date_bytes = [0 as u8; 6];
        for i in 0..6 {
            expiry_date_bytes[i] = mrz[ID_CARD_MRZ_EXPIRY_DATE_INDEX + i];
        }
        Date::from_bytes_short_year(expiry_date_bytes, threshold_year)
    } else {
        let mut expiry_date_bytes = [0 as u8; 6];
        for i in 0..6 {
            expiry_date_bytes[i] = mrz[PASSPORT_MRZ_EXPIRY_DATE_INDEX + i];
        }
        Date::from_bytes_short_year(expiry_date_bytes, threshold_year)
    };

    assert(current_date.lt(expiry_date), "Document is expired");
}
