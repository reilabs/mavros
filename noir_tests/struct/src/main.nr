pub struct IDData {
    pub lucky_numbers: [u8; 2],
    pub age: u8,
}


fn main(x: IDData) {
    assert_eq(x.age, x.lucky_numbers[0] * x.lucky_numbers[1]);
}