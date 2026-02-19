use std::hash::poseidon2_permutation;
use poseidon2::bn254::perm;

fn main(x: Field) {
    let array4 = [x, x, x, x];
    let swapped4 = poseidon2_permutation(array4, 4);
    let library4 = perm::x5_4(array4);
    for i in 0..4 {
        assert_eq(swapped4[i], library4[i]);
    }
    
    let array8 = [x, x, x, x, x, x, x, x];
    let swapped8 = poseidon2_permutation(array8, 8);
    let library8 = perm::x5_8(array8);
    
    for i in 0..8 {
        assert_eq(swapped8[i], library8[i]);
    }


}
