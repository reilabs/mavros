// Inserting past `len` at a witness index must trap at witness generation (the OOB bound is a
// runtime property of the witness input).
fn main(i: u32) {
    let s: [Field] = [10];
    let s2 = s.insert(i, 1);
    assert_eq(s2.len() as Field, 2);
}
