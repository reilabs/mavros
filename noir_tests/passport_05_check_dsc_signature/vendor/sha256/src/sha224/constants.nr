use crate::sha256::constants::STATE;
pub(crate) global INITIAL_STATE_SHA224: STATE =
    [3238371032, 914150663, 812702999, 4144912697, 4290775857, 1750603025, 1694076839, 3204075428];

// The hash is 28 bytes.
pub(crate) type HASH_SHA224 = [u8; 28];
