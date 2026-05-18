// Reproducer from https://github.com/reilabs/mavros/issues/165.
//
// Iteratively hashing a witness `[u8; 32]` through `sha256_var` exercises:
//   - the witness-cast no-op path (#165): the `data` Jmp arg into the loop
//     head carries a tuple-shaped value whose pure u32 fields used to trip
//     `emit_value_conversion`,
//   - witness `Shl/Shr` lowering: sha256's byte-packing in
//     `process_full_blocks` / `finalize_sha256_blocks` shifts witness u32s by
//     pure amounts,
//   - witness big-endian `ToRadix` lowering: `(state[j] as Field).to_be_bytes()`
//     in `hash_final_block`,
//   - the `Radix::Dyn -> Bytes` rewrite: `to_be_bytes` reaches the builtin
//     through `to_be_radix(256)` so the radix is a stdlib function parameter
//     that gets runtime-asserted == 256.
use sha256::sha256_var;

global NUM_SHA_CALLS: u32 = 35;

fn main(input: [u8; 32]) {
    let mut data = input;
    for _ in 0..NUM_SHA_CALLS {
        data = sha256_var(data, 32);
    }

    // SHA-256 chain applied 35 times to [0, 1, .., 31]; computed offline.
    let expected: [u8; 32] = [
        189, 242, 164, 187, 21, 119, 3, 146,
        78, 16, 192, 123, 166, 232, 109, 185,
        207, 129, 243, 47, 161, 30, 47, 173,
        31, 7, 141, 31, 9, 4, 134, 245,
    ];
    for i in 0..32 {
        assert(data[i] == expected[i]);
    }
}
