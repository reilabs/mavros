fn main(x: Field, flag: Field) {
    if flag == 1 {
        let val: u8 = x as u8;
        assert(val as Field == x);
    }
}
