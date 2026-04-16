use sha256::sha256_var;
use utils::check_zero_padding;

pub fn sha256_and_check_data_to_sign<let DATA_TO_SIGN_MAX_LEN: u32>(
    data_to_sign: [u8; DATA_TO_SIGN_MAX_LEN],
    data_to_sign_len: u32,
) -> [u8; 32] {
    check_zero_padding(data_to_sign, data_to_sign_len);
    sha256_var(data_to_sign, data_to_sign_len as u64)
}
