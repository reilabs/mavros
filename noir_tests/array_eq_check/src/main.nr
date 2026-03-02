// Tests comparing two byte arrays element-by-element.
// Needed by: passport hash verification - comparing computed SHA-256
//            hash against expected hash stored in SOD/eContent.
// Pattern: for i in 0..N { assert_eq(computed[i], expected[i]); }
fn main(hash: [u8; 8], data: [u8; 16], offset: u32) {
    // Extract subarray from data at offset and compare with hash
    for i in 0..8 {
        assert_eq(data[offset + i], hash[i]);
    }
}
