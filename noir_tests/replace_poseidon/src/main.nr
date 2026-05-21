use std::hash::poseidon2_permutation;
use poseidon2::bn254::permutation;

fn main(x: Field) {
    let swapped = poseidon2_permutation([x, x, x, x]);
    let library = permutation::t4([x, x, x, x]);
    for i in 0..4 {
        assert_eq(swapped[i], library[i]);
    }
}
