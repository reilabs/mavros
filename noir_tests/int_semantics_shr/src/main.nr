// GENERATED FILE -- do not edit by hand.
//
// Rendered by `mavros-int-semantics`'s `corpus` module from the model itself. Regenerate with
// `MAVROS_BLESS=1 cargo test -p mavros-int-semantics`, and read the diff: a program whose
// expected answer moved is a program whose *meaning* moved.
//
// Every operand here is a `main` parameter, and therefore a witness. That is the point: a
// constant pair would be folded before the guards, the witness lowering, the VM and the WASM
// backend ever saw it, and those are the five evaluators a unit test cannot reach. The
// `assert_eq` then makes each answer observable to the R1CS-satisfaction oracle in every lane.
//
// Operation: `>>`. Cells, as (reading, width, pairs):
//   * unsigned, 8 bits, 6 pair(s)
//   * unsigned, 16 bits, 6 pair(s)
//   * unsigned, 32 bits, 6 pair(s)
//   * unsigned, 64 bits, 6 pair(s)
//   * unsigned, 128 bits, 6 pair(s)
//   * signed, 8 bits, 6 pair(s)
//   * signed, 16 bits, 6 pair(s)
//   * signed, 32 bits, 6 pair(s)
//   * signed, 64 bits, 6 pair(s)

fn main(
    u8_0: u8,
    u8_1: u8,
    u8_2: u8,
    u8_3: u8,
    u8_4: u8,
    u8_5: u8,
    u8_6: u8,
    u8_7: u8,
    u8_8: u8,
    u8_9: u8,
    u16_0: u16,
    u16_1: u16,
    u16_2: u16,
    u16_3: u16,
    u16_4: u16,
    u16_5: u16,
    u16_6: u16,
    u16_7: u16,
    u16_8: u16,
    u16_9: u16,
    u16_10: u16,
    u32_0: u32,
    u32_1: u32,
    u32_2: u32,
    u32_3: u32,
    u32_4: u32,
    u32_5: u32,
    u32_6: u32,
    u32_7: u32,
    u32_8: u32,
    u32_9: u32,
    u32_10: u32,
    u64_0: u64,
    u64_1: u64,
    u64_2: u64,
    u64_3: u64,
    u64_4: u64,
    u64_5: u64,
    u64_6: u64,
    u64_7: u64,
    u64_8: u64,
    u64_9: u64,
    u64_10: u64,
    u128_0: u128,
    u128_1: u128,
    u128_2: u128,
    u128_3: u128,
    u128_4: u128,
    u128_5: u128,
    u128_6: u128,
    u128_7: u128,
    u128_8: u128,
    u128_9: u128,
    u128_10: u128,
    i8_0: i8,
    i8_1: i8,
    i8_2: i8,
    i8_3: i8,
    i8_4: i8,
    i8_5: i8,
    i8_6: i8,
    i8_7: i8,
    i8_8: i8,
    i8_9: i8,
    i16_0: i16,
    i16_1: i16,
    i16_2: i16,
    i16_3: i16,
    i16_4: i16,
    i16_5: i16,
    i16_6: i16,
    i16_7: i16,
    i16_8: i16,
    i16_9: i16,
    i16_10: i16,
    i32_0: i32,
    i32_1: i32,
    i32_2: i32,
    i32_3: i32,
    i32_4: i32,
    i32_5: i32,
    i32_6: i32,
    i32_7: i32,
    i32_8: i32,
    i32_9: i32,
    i32_10: i32,
    i64_0: i64,
    i64_1: i64,
    i64_2: i64,
    i64_3: i64,
    i64_4: i64,
    i64_5: i64,
    i64_6: i64,
    i64_7: i64,
    i64_8: i64,
    i64_9: i64,
    i64_10: i64,
) {
    // unsigned at 8 bits: 6 pair(s).
    assert_eq(u8_0 >> u8_0, 0);
    assert_eq(u8_2 >> u8_5, 0);
    assert_eq(u8_6 >> u8_4, 1);
    assert_eq(u8_7 >> u8_3, 7);
    assert_eq(u8_8 >> u8_2, 32);
    assert_eq(u8_9 >> u8_1, 126);

    // unsigned at 16 bits: 6 pair(s).
    assert_eq(u16_0 >> u16_0, 0);
    assert_eq(u16_2 >> u16_6, 0);
    assert_eq(u16_7 >> u16_5, 0);
    assert_eq(u16_8 >> u16_4, 85);
    assert_eq(u16_9 >> u16_3, 256);
    assert_eq(u16_10 >> u16_1, 32766);

    // unsigned at 32 bits: 6 pair(s).
    assert_eq(u32_0 >> u32_0, 0);
    assert_eq(u32_2 >> u32_6, 0);
    assert_eq(u32_7 >> u32_5, 0);
    assert_eq(u32_8 >> u32_4, 21845);
    assert_eq(u32_9 >> u32_3, 65536);
    assert_eq(u32_10 >> u32_1, 2147483646);

    // unsigned at 64 bits: 6 pair(s).
    assert_eq(u64_0 >> u64_0, 0);
    assert_eq(u64_2 >> u64_6, 0);
    assert_eq(u64_7 >> u64_5, 0);
    assert_eq(u64_8 >> u64_4, 1431655765);
    assert_eq(u64_9 >> u64_3, 4294967296);
    assert_eq(u64_10 >> u64_1, 9223372036854775806);

    // unsigned at 128 bits: 6 pair(s).
    assert_eq(u128_0 >> u128_0, 0);
    assert_eq(u128_2 >> u128_7, 0);
    assert_eq(u128_6 >> u128_5, 0);
    assert_eq(u128_8 >> u128_4, 6148914691236517205);
    assert_eq(u128_9 >> u128_3, 18446744073709551616);
    assert_eq(u128_10 >> u128_1, 170141183460469231731687303715884105726);

    // signed at 8 bits: 6 pair(s).
    assert_eq(i8_2 >> i8_2, 0);
    assert_eq(i8_4 >> i8_7, 0);
    assert_eq(i8_8 >> i8_6, 1);
    assert_eq(i8_9 >> i8_5, 7);
    assert_eq(i8_0 >> i8_4, -32);
    assert_eq(i8_1 >> i8_3, -2);

    // signed at 16 bits: 6 pair(s).
    assert_eq(i16_2 >> i16_2, 0);
    assert_eq(i16_4 >> i16_8, 0);
    assert_eq(i16_9 >> i16_7, 0);
    assert_eq(i16_10 >> i16_6, 85);
    assert_eq(i16_0 >> i16_5, -256);
    assert_eq(i16_1 >> i16_3, -2);

    // signed at 32 bits: 6 pair(s).
    assert_eq(i32_2 >> i32_2, 0);
    assert_eq(i32_4 >> i32_8, 0);
    assert_eq(i32_9 >> i32_7, 0);
    assert_eq(i32_10 >> i32_6, 21845);
    assert_eq(i32_0 >> i32_5, -65536);
    assert_eq(i32_1 >> i32_3, -2);

    // signed at 64 bits: 6 pair(s).
    assert_eq(i64_2 >> i64_2, 0);
    assert_eq(i64_4 >> i64_8, 0);
    assert_eq(i64_9 >> i64_7, 0);
    assert_eq(i64_10 >> i64_6, 1431655765);
    assert_eq(i64_0 >> i64_5, -4294967296);
    assert_eq(i64_1 >> i64_3, -2);
}
