fn main(x: Field, flag: Field) {
    // flag is a witness-valued boolean (0 or 1)
    // When flag == 1, we rangecheck x as u8 (0..255)
    // When flag == 0, x can be anything (rangecheck is skipped)
    if flag == 1 {
        let val: u8 = x as u8;
        assert(val as Field == x);
    }
}
