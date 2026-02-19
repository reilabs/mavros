mod consts;
pub mod permutation;

pub struct Poseidon2Bn254Config<let T: u32, let R: u32> {
    first_full_rc: [[Field; T]; 4],
    partial_rc: [Field; R],
    second_full_rc: [[Field; T]; 4],
}

fn permute_bn254<let T: u32, let R: u32>(
    mut state: [Field; T],
    mm_external: fn([Field; T]) -> [Field; T],
    mm_internal: fn([Field; T]) -> [Field; T],
    config: Poseidon2Bn254Config<T, R>,
) -> [Field; T] {
    state = mm_external(state);
    for r in 0..4 {
        state = crate::vec_add(state, config.first_full_rc[r]);
        state = crate::sbox(state);
        state = mm_external(state);
    }

    for r in 0..R {
        state[0] += config.partial_rc[r];
        state[0] = crate::sbox_e(state[0]);
        state = mm_internal(state);
    }
    for r in 0..4 {
        state = crate::vec_add(state, config.second_full_rc[r]);
        state = crate::sbox(state);
        state = mm_external(state);
    }
    state
}
