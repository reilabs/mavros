use poseidon::poseidon::bn254;

fn main(input: u8, out: pub [u8; 32]) {
    let mut input = input as Field;
    let out = Field::from_le_bytes_checked(out);

    for i in 0..10000 {
      input = bn254::hash_1([input]);
    }
    assert_eq(input, out);
}
