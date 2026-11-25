fn main(x: [u8; 8], flag: u8) {
    // Read from bytes, so the element is witnessed with no range the compiler tracks, and its
    // narrowing casts truncate.
    let x = Field::from_le_bytes_checked(x);
    let flag = flag as Field;

    if flag == 1 {
        let val: u8 = x as u8;
        assert(val as Field == x);
    }
}
