pub struct IDData {
    pub lucky_numbers: [Field; 2],
    pub age: Field,
}


fn main(x: IDData) {
    assert_eq(x.age, x.lucky_numbers[0] * x.lucky_numbers[1]);
}