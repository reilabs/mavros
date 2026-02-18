use std::hash::poseidon2_permutation;
use poseidon2::bn254::perm;

fn main(x: [Field; 4]) {
    let swapped = poseidon2_permutation(x, 4);
    let library = perm::x5_4(x);
    for i in 0..4 {
        assert_eq(swapped[i], library[i]);
    }
}
