use std::hash::poseidon2_permutation;

fn main(a: Field, b: Field, c: Field, d: Field) {
    let state = [a, b, c, d];
    let result = poseidon2_permutation(state, 4);
    assert(result[0] != 0);
}
