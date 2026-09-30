// A witnessed `u128` shifted by a pure amount a loop reaches the width with must be rejected.
//
// The amount is known wherever the constraints are built, so the limb-wise shift checks it with a
// comparison rather than by witnessing its decomposition, and the last iteration's is false.
fn main(x: u128) {
    let mut sum = 0;
    for i in 0..5 {
        sum += x >> ((i * 32) as u128);
    }
    assert(sum != 1);
}
