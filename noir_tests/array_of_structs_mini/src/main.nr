pub struct IDData {
    pub age: Field,
}


fn main(x: [IDData; 1]) {
    assert_eq(x[0].age, 18);
}