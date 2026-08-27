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
// Operation: `^`. Cells, as (reading, width, pairs):
//   * unsigned, 8 bits, 3 pair(s)
//   * unsigned, 16 bits, 3 pair(s)
//   * unsigned, 32 bits, 3 pair(s)
//   * unsigned, 64 bits, 3 pair(s)
//   * unsigned, 128 bits, 3 pair(s)
//   * signed, 8 bits, 3 pair(s)
//   * signed, 16 bits, 3 pair(s)
//   * signed, 32 bits, 3 pair(s)
//   * signed, 64 bits, 3 pair(s)

fn main(
    u8_0: u8,
    u8_1: u8,
    u8_2: u8,
    u16_0: u16,
    u16_1: u16,
    u16_2: u16,
    u32_0: u32,
    u32_1: u32,
    u32_2: u32,
    u64_0: u64,
    u64_1: u64,
    u64_2: u64,
    u128_0: u128,
    u128_1: u128,
    u128_2: u128,
    i8_0: i8,
    i8_1: i8,
    i8_2: i8,
    i16_0: i16,
    i16_1: i16,
    i16_2: i16,
    i32_0: i32,
    i32_1: i32,
    i32_2: i32,
    i64_0: i64,
    i64_1: i64,
    i64_2: i64,
) {
    // unsigned at 8 bits: 3 pair(s).
    assert_eq(u8_0 ^ u8_0, 0);
    assert_eq(u8_1 ^ u8_2, 169);
    assert_eq(u8_2 ^ u8_1, 169);

    // unsigned at 16 bits: 3 pair(s).
    assert_eq(u16_0 ^ u16_0, 0);
    assert_eq(u16_1 ^ u16_2, 32868);
    assert_eq(u16_2 ^ u16_1, 32868);

    // unsigned at 32 bits: 3 pair(s).
    assert_eq(u32_0 ^ u32_0, 0);
    assert_eq(u32_1 ^ u32_2, 2147483748);
    assert_eq(u32_2 ^ u32_1, 2147483748);

    // unsigned at 64 bits: 3 pair(s).
    assert_eq(u64_0 ^ u64_0, 0);
    assert_eq(u64_1 ^ u64_2, 9223372036854775908);
    assert_eq(u64_2 ^ u64_1, 9223372036854775908);

    // unsigned at 128 bits: 3 pair(s).
    assert_eq(u128_0 ^ u128_0, 0);
    assert_eq(u128_1 ^ u128_2, 170141183460469231731687303715884105828);
    assert_eq(u128_2 ^ u128_1, 170141183460469231731687303715884105828);

    // signed at 8 bits: 3 pair(s).
    assert_eq(i8_1 ^ i8_1, 0);
    assert_eq(i8_2 ^ i8_0, -87);
    assert_eq(i8_0 ^ i8_2, -87);

    // signed at 16 bits: 3 pair(s).
    assert_eq(i16_1 ^ i16_1, 0);
    assert_eq(i16_2 ^ i16_0, -32668);
    assert_eq(i16_0 ^ i16_2, -32668);

    // signed at 32 bits: 3 pair(s).
    assert_eq(i32_1 ^ i32_1, 0);
    assert_eq(i32_2 ^ i32_0, -2147483548);
    assert_eq(i32_0 ^ i32_2, -2147483548);

    // signed at 64 bits: 3 pair(s).
    assert_eq(i64_1 ^ i64_1, 0);
    assert_eq(i64_2 ^ i64_0, -9223372036854775708);
    assert_eq(i64_0 ^ i64_2, -9223372036854775708);
}
