pub struct IDData {
    pub age: u8,
}


fn main(x: [IDData; 1]) {
    assert_eq(x[0].age, 18);
}