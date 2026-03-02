// Tests a struct with byte array fields as main() input parameter.
// Needed by: PassportValidityContents struct which has fields like
//   econtent: [u8; 200], dsc_signature: [u8; 256], etc.
struct CertData {
    hash: [u8; 32],
    hash_offset: u32,
    data: [u8; 64],
    data_len: u32,
}

fn main(cert: CertData) {
    // Verify hash is at the expected offset in data
    assert(cert.hash_offset + 32 <= cert.data_len);

    for i in 0..32 {
        assert_eq(cert.data[cert.hash_offset + i], cert.hash[i]);
    }
}
