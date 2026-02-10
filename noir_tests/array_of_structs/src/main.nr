pub struct IDData {
    pub age: Field,
    pub id: Field,
}


fn main(x: [IDData; 2]) {
    assert_eq(x[1].age, 18);
}