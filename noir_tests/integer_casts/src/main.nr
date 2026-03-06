// Tests casting between u8, u32, u64, u128, and Field.
// Needed by: passport code casts extensively between integer types
//   - compare_age: dg1[i] as u8, values as u32
//   - utils: x as Field, x as u128
//   - sig_check: values as u64
fn main(byte_val: u8, word_val: u32) {
    // u8 -> u32 (widening)
    let w: u32 = byte_val as u32;
    assert_eq(w, 200);

    // u32 -> u64 (widening, needed for wrapping add)
    let q: u64 = word_val as u64;
    assert_eq(q, 1000);

    // u8 -> Field (used in byte packing)
    let f: Field = byte_val as Field;
    assert_eq(f, 200);

    // u32 -> Field
    let f2: Field = word_val as Field;
    assert_eq(f2, 1000);

    // u64 -> u32 (narrowing, needed for wrapping add result)
    let big_val: u64 = 300;
    let narrow: u32 = big_val as u32;
    assert_eq(narrow, 300);

    // u8 -> u64
    let byte_as_u64: u64 = byte_val as u64;
    assert_eq(byte_as_u64, 200);

    // u64 -> u128 (for bignum packing)
    let val64: u64 = 12345;
    let val128: u128 = val64 as u128;
    assert_eq(val128, 12345);
}
