use date::Date;

global DG1_TO_MRZ_OFFSET: u32 = 5;
global ID_CARD_MRZ_EXPIRY_DATE_INDEX: u32 = 38;
global PASSPORT_MRZ_EXPIRY_DATE_INDEX: u32 = 65;

fn get_array_slice<let N: u32, let M: u32>(arr: [u8; N], start: u32, end: u32) -> [u8; M] {
    let mut slice = [0 as u8; M];
    for i in start..end {
        slice[i - start] = arr[i];
    }
    slice
}

fn is_id_card(dg1: [u8; 95]) -> bool {
    (dg1[93] != 0) & (dg1[94] != 0)
}

fn get_mrz_from_dg1(dg1: [u8; 95]) -> [u8; 90] {
    let mut mrz = [0 as u8; 90];
    for i in 0..90 {
        mrz[i] = dg1[i + DG1_TO_MRZ_OFFSET];
    }
    mrz
}

fn main(
    dg1: [u8; 95],
    current_date: pub u64,
) {
    let current_date = Date::from_timestamp(current_date);
    let threshold_year = current_date.add_years(30);
    let mrz = get_mrz_from_dg1(dg1);

    let expiry_date = if is_id_card(dg1) {
        let expiry_date_bytes = get_array_slice(
            mrz,
            ID_CARD_MRZ_EXPIRY_DATE_INDEX,
            ID_CARD_MRZ_EXPIRY_DATE_INDEX + 6,
        );
        Date::from_bytes_short_year(expiry_date_bytes, threshold_year)
    } else {
        let expiry_date_bytes = get_array_slice(
            mrz,
            PASSPORT_MRZ_EXPIRY_DATE_INDEX,
            PASSPORT_MRZ_EXPIRY_DATE_INDEX + 6,
        );
        Date::from_bytes_short_year(expiry_date_bytes, threshold_year)
    };

    assert(current_date.lt(expiry_date), "Document is expired");
}
