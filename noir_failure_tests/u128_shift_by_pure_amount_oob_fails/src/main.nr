// A witnessed `u128` shifted by a pure loop amount that reaches the width must be rejected. The
// limb-wise shift checks a pure amount by comparison, which fails on the last iteration.
fn main(x: u128) {
    let mut sum = 0;
    for i in 0..5 {
        sum += x >> ((i * 32) as u128);
    }
    assert(sum != 1);
}
