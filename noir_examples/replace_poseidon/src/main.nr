use std::hash::poseidon2_permutation;
use poseidon2::bn254::perm;

fn main(x: Field) {
    let swapped2 = poseidon2_permutation([x, x], 2);
    let library2 = perm::x5_2([x, x]);
    for i in 0..2 {
        assert_eq(swapped2[i], library2[i]);
    }

    let swapped4 = poseidon2_permutation([x, x, x, x], 4);
    let library4 = perm::x5_4([x, x, x, x]);

    for i in 0..4 {
        assert_eq(swapped4[i], library4[i]);
    }
}
