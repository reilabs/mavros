use std::hash::blake2s;

fn prefix<let N: u32>(msg: [u8; 128]) -> [u8; N] {
    let mut out: [u8; N] = [0; N];
    for i in 0..N {
        out[i] = msg[i];
    }
    out
}

// BLAKE2s over prefixes of the KAT-style message 00 01 02 ... 7f at lengths
// around the 64-byte block boundary, plus a digest-of-digest to exercise
// chaining through a witness input. Expected digests come from the `blake2`
// crate acvm pins.
// (https://github.com/BLAKE2/BLAKE2/blob/master/testvectors/blake2-kat.json).

fn main(
    msg: [u8; 128],
    expected_0: pub [u8; 32],
    expected_3: pub [u8; 32],
    expected_63: pub [u8; 32],
    expected_64: pub [u8; 32],
    expected_65: pub [u8; 32],
    expected_128: pub [u8; 32],
    expected_chained: pub [u8; 32],
) {
    let empty: [u8; 0] = [];
    assert_eq(blake2s(empty), expected_0);

    let m3: [u8; 3] = prefix(msg);
    assert_eq(blake2s(m3), expected_3);
    let m63: [u8; 63] = prefix(msg);
    assert_eq(blake2s(m63), expected_63);
    let m64: [u8; 64] = prefix(msg);
    assert_eq(blake2s(m64), expected_64);
    let m65: [u8; 65] = prefix(msg);
    assert_eq(blake2s(m65), expected_65);

    let full = blake2s(msg);
    assert_eq(full, expected_128);
    assert_eq(blake2s(full), expected_chained);
}
