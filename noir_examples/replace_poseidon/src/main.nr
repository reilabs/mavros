use std::hash::poseidon2_permutation;
use poseidon2::bn254::perm;

fn main(x: Field) {
    let array2 = [x, x];
    let swapped2 = poseidon2_permutation(array2, 2);
    let library2 = perm::x5_2(array2);
    for i in 0..2 {
        assert_eq(swapped2[i], library2[i]);
    }
    
    let array4 = [x, x, x, x];
    let swapped4 = poseidon2_permutation(array4, 4);
    let library4 = perm::x5_4(array4);
    
    for i in 0..4 {
        assert_eq(swapped4[i], library4[i]);
    }
}
