fn from_le_bits<let N: u32>(bits: [bool; N]) -> Field {
    let mut result = 0;
    let mut coefficient = 1;
    for i in 0..N {
        result += (bits[i] as Field) * coefficient;
        coefficient *= 2;
    }
    result
}

fn from_be_bits<let N: u32>(bits: [bool; N]) -> Field {
    let mut result = 0;
    let mut coefficient = 1;
    for i in 0..N {
        result += (bits[N - 1 - i] as Field) * coefficient;
        coefficient *= 2;
    }
    result
}

fn main(x: Field) {
    // Exercise the maximum supported BN254 width through both native bytecode operations.
    let le: [bool; 254] = x.to_le_bits();
    let be: [bool; 254] = x.to_be_bits();

    assert_eq(from_le_bits(le), x);
    assert_eq(from_be_bits(be), x);
    for i in 0..254 {
        assert_eq(le[i], be[253 - i]);
    }

    // Keep an explicit ordering check so matching-but-reversed implementations cannot agree by
    // accident in the roundtrip assertions above.
    let small: Field = 13;
    let small_le: [bool; 4] = small.to_le_bits();
    let small_be: [bool; 4] = small.to_be_bits();
    assert_eq(small_le, [true, false, true, true]);
    assert_eq(small_be, [true, true, false, true]);
}
