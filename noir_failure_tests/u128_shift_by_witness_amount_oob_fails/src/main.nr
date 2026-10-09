// A witnessed `u128` shift by an amount equal to the width must be rejected.
//
// The limb-wise shift splits its amount into a number of whole 64-bit limbs and a remainder, and at
// 128 bits that split alone bounds it: one bit of whole limbs plus a remainder below 64 cannot
// reach 128.
fn main(x: u128, n: u128) {
    let r = x << n;
    assert(r == 0);
}
