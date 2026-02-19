use crate::bn254::consts;
use crate::bn254::permute_bn254;

fn internal_2(state: [Field; 2]) -> [Field; 2] {
    let sum = state[0] + state[1];
    [sum + state[0], sum + 2 * state[1]]
}

fn internal_3(state: [Field; 3]) -> [Field; 3] {
    let sum = state.reduce(|a, b| a + b);
    let mut ret_val = [0; 3];
    ret_val[0] = state[0] + sum;
    ret_val[1] = state[1] + sum;
    ret_val[2] = 2 * state[2] + sum;
    ret_val
}

fn external_3(mut state: [Field; 3]) -> [Field; 3] {
    let sum = state.reduce(|a, b| a + b);
    state[0] += sum;
    state[1] += sum;
    state[2] += sum;
    state
}

fn external<let T: u32>(mut state: [Field; T]) -> [Field; T] {
    let mut mds_parts: [[Field; 4]; T / 4] = [[0; 4]; T / 4];
    for i in 0..T / 4 {
        let offset = (4 * i);
        mds_parts[i] = crate::mds_4([
            state[0 + offset],
            state[1 + offset],
            state[2 + offset],
            state[3 + offset],
        ]);
    }
    for i in 0..T / 4 {
        for j in 0..4 {
            state[i * 4 + j] = mds_parts[i][j];
        }
    }

    let mut stored = [0; 4];
    for l in 0..4 {
        stored[l] = mds_parts[0][l];
        for j in 1..T / 4 {
            stored[l] += mds_parts[j][l];
        }
    }
    for i in 0..T / 4 {
        for j in 0..4 {
            state[i * 4 + j] += stored[j];
        }
    }

    state
}

fn internal_4(mut state: [Field; 4]) -> [Field; 4] {
    diag_mat_mul(state, consts::internal_diag4)
}

fn internal_8(mut state: [Field; 8]) -> [Field; 8] {
    diag_mat_mul(state, consts::internal_diag8)
}

fn internal_12(mut state: [Field; 12]) -> [Field; 12] {
    diag_mat_mul(state, consts::internal_diag12)
}

fn internal_16(mut state: [Field; 16]) -> [Field; 16] {
    diag_mat_mul(state, consts::internal_diag16)
}

fn diag_mat_mul<let T: u32>(mut state: [Field; T], diag: [Field; T]) -> [Field; T] {
    let sum = state.reduce(|a, b| a + b);
    for i in 0..T {
        state[i] *= diag[i];
        state[i] += sum;
    }
    state
}

#[field(bn254)]
pub fn t2(input: [Field; 2]) -> [Field; 2] {
    permute_bn254(input, crate::mds_2, internal_2, consts::x5_2_config())
}

#[field(bn254)]
pub fn t3(input: [Field; 3]) -> [Field; 3] {
    permute_bn254(input, external_3, internal_3, consts::x5_3_config())
}

#[field(bn254)]
pub fn t4(input: [Field; 4]) -> [Field; 4] {
    permute_bn254(input, crate::mds_4, internal_4, consts::x5_4_config())
}

#[field(bn254)]
pub fn t8(input: [Field; 8]) -> [Field; 8] {
    permute_bn254(input, external, internal_8, consts::x5_8_config())
}

#[field(bn254)]
pub fn t12(input: [Field; 12]) -> [Field; 12] {
    permute_bn254(input, external, internal_12, consts::x5_12_config())
}

#[field(bn254)]
pub fn t16(input: [Field; 16]) -> [Field; 16] {
    permute_bn254(input, external, internal_16, consts::x5_16_config())
}
