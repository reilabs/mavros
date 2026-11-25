pub struct IDData {
    pub age: u8,
    pub id: u8,
}


fn main(x: [IDData; 2]) {
    assert_eq(x[1].age, 18);
}