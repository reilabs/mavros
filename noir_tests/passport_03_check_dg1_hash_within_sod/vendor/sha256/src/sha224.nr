use crate::sha256::constants::HASH;

use crate::sha256::{__sha_var, finalize_sha256_blocks, partial_sha256_var_end, process_full_blocks};

use constants as sha224_constants;

pub(crate) mod constants;
mod tests;

// Variable size SHA-224 hash
pub fn sha224_var<let N: u32>(msg: [u8; N], message_size: u64) -> sha224_constants::HASH_SHA224 {
    let message_size = message_size as u32;
    assert(message_size <= N);

    if std::runtime::is_unconstrained() {
        // Safety: SHA224 is running as an unconstrained function.
        unsafe {
            __sha224_var(msg, message_size)
        }
    } else {
        let (mut h, mut msg_block) =
            process_full_blocks(msg, message_size, sha224_constants::INITIAL_STATE_SHA224);

        let hash: HASH = finalize_sha256_blocks::<N>(message_size, h, msg_block);
        // Convert 32-byte hash to 28-byte hash for SHA-224
        let mut hash_sha224: sha224_constants::HASH_SHA224 = [0; 28];
        for i in 0..28 {
            hash_sha224[i] = hash[i];
        }
        hash_sha224
    }
}

// Variable size SHA-224 hash
unconstrained fn __sha224_var<let N: u32>(
    msg: [u8; N],
    message_size: u32,
) -> sha224_constants::HASH_SHA224 {
    let hash = __sha_var(msg, message_size, sha224_constants::INITIAL_STATE_SHA224);
    let mut hash_sha224: sha224_constants::HASH_SHA224 = [0; 28];
    for i in 0..28 {
        hash_sha224[i] = hash[i];
    }
    hash_sha224
}

/**
 * Given some state of a partially computed sha256/sha224 hash and remaining preimage, complete the hash
 * @notice used for traditional partial hashing
 *
 * @param N - the maximum length of the message to hash
 * @param h - the intermediate hash state
 * @param msg - the remaining preimage to hash
 * @param message_size - the size of the current chunk
 * @param real_message_size - the total size of the original preimage
 * @return finalized sha24 hash
 */
pub fn partial_sha224_var_end<let N: u32>(
    mut h: [u32; 8],
    msg: [u8; N],
    message_size: u32,
    real_message_size: u32,
) -> sha224_constants::HASH_SHA224 {
    let hash = partial_sha256_var_end(h, msg, message_size, real_message_size);
    let mut hash_sha224: sha224_constants::HASH_SHA224 = [0; 28];
    for i in 0..28 {
        hash_sha224[i] = hash[i];
    }
    hash_sha224
}

mod equivalence_test {

    #[test]
    fn test_implementations_agree(msg: [u8; 100], message_size: u64) {
        let message_size = message_size % 100;
        // Safety: testing
        let unconstrained_sha224 = unsafe { super::__sha224_var(msg, message_size as u32) };
        let sha224 = super::sha224_var(msg, message_size);
        assert_eq(sha224, unconstrained_sha224);
    }
}
