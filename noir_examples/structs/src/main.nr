pub struct IDData {
    pub id: u8,
    pub lucky_number: u8,
}


fn main(x: [IDData; 1]) {
    assert(x[0].id < 100, "ID must be less than 100");
}