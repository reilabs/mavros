pub struct IDData {
    pub age: Field,
    pub ids: [Field; 2],
}


fn main(x: [IDData; 2]) {
    assert_eq(x[1].ids[0], 18);
}