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

fn main(
    _dg1: [u8; 95],
    _dg1_padded_length: u64,
    _current_date: pub u64,
    _min_age_required: pub u8,
    _max_age_required: pub u8,
    passport_validity_contents: PassportValidityContents,
) {
    assert(
        passport_validity_contents.dsc_pubkey_offset_in_dsc_cert + DSC_SIGNATURE_SIZE <= DSC_CERT_MAX_SIZE,
        "pubkey_offset + DSC_KEY_SIZE exceeds TBS certificate size",
    );
    for i in 0..DSC_SIGNATURE_SIZE {
        assert(
            passport_validity_contents.dsc_cert[i + passport_validity_contents.dsc_pubkey_offset_in_dsc_cert]
                == passport_validity_contents.dsc_pubkey[i],
        );
    }
}
