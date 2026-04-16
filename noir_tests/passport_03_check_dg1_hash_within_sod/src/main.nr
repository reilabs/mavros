use sha256::sha256_var;

global SIGNED_ATTRIBUTES_MAX_SIZE: u32 = 200;
global DSC_CERT_MAX_SIZE: u32 = 1300;
global DSC_SIGNATURE_SIZE: u32 = 256;
global CSC_PUBKEY_SIZE: u32 = 512;

pub struct PassportValidityContents {
    pub signed_attributes: [u8; SIGNED_ATTRIBUTES_MAX_SIZE],
    pub signed_attributes_size: u64,
    pub dsc_signature: [u8; DSC_SIGNATURE_SIZE],
    pub dsc_rsa_exponent: u32,
    pub dsc_pubkey: [u8; DSC_SIGNATURE_SIZE],
    pub dsc_barrett_mu: [u8; DSC_SIGNATURE_SIZE + 1],
    pub dsc_pubkey_offset_in_dsc_cert: u32,
    pub dsc_cert: [u8; DSC_CERT_MAX_SIZE],
    pub dsc_cert_len: u32,
    pub csc_pubkey: [u8; CSC_PUBKEY_SIZE],
    pub csc_barrett_mu: [u8; CSC_PUBKEY_SIZE + 1],
    pub dsc_cert_signature: [u8; CSC_PUBKEY_SIZE],
    pub csc_rsa_exponent: u32,
    pub econtent: [u8; SIGNED_ATTRIBUTES_MAX_SIZE],
    pub econtent_len: u32,
    pub dg1_hash_offset: u32,
    pub econtent_hash_offset: u32,
}

fn check_zero_padding<let N: u32>(padded_array: [u8; N], len: u32) {
    assert(len <= N, "check_zero_padding: len exceeds array size");
    for i in 0..N {
        if i >= len {
            assert_eq(padded_array[i], 0);
        }
    }
}

fn main(
    dg1: [u8; 95],
    dg1_padded_length: u64,
    _current_date: pub u64,
    _min_age_required: pub u8,
    _max_age_required: pub u8,
    passport_validity_contents: PassportValidityContents,
) {
    check_zero_padding(
        passport_validity_contents.econtent,
        passport_validity_contents.econtent_len,
    );
    assert(
        passport_validity_contents.dg1_hash_offset + 32 <= passport_validity_contents.econtent_len,
        "dg1_hash_offset out of bounds",
    );
    assert(dg1_padded_length <= 95, "dg1_padded_length out of bounds");

    let dg1_hash = sha256_var(dg1, dg1_padded_length);
    for i in 0..32 {
        assert(
            dg1_hash[i]
                == passport_validity_contents.econtent[passport_validity_contents.dg1_hash_offset + i],
        );
    }
}
