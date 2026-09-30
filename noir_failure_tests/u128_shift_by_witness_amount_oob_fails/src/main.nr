// A witnessed `u128` shift by an amount at the width must be rejected.
//
// The limb-wise shift reads its amount as a whole number of 64-bit limbs and the rest, and at 128
// bits that decomposition is its own bound: a single bit of whole limbs and a remainder below 64
// cannot spell 128.
fn main(x: u128, n: u128) {
    let r = x << n;
    assert(r == 0);
}
