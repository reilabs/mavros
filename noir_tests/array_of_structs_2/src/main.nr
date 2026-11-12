pub struct IDData {
    pub age: u8,
    pub ids: [u8; 2],
}


fn main(x: [IDData; 2]) {
    assert_eq(x[1].ids[0], 18);
}