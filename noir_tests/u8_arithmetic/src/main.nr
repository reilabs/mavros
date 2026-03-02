// Tests basic u8 arithmetic on witness (non-constant) values.
// Needed by: MRZ byte parsing in passport: dg1[i] - 48 to convert
//            ASCII digit to number, date arithmetic on u8 values.
// This is the most fundamental integer arithmetic test. The R1CS
// generator currently only handles integer arithmetic on constants.
fn main(a: u8, b: u8, expected: pub Field) {
    // u8 addition
    let sum = a + b;
    assert_eq(sum as Field, expected);
}
