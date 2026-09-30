// Witnessed `u128` shifts, which no single field product can hold: `x << n` reaches `2^255` for an
// amount up to 127, past the bn254 modulus. `WideWitnessInts` lowers each limb-wise, apart from a
// `>>` by a literal, which the simplifier has already made a bit window.
//
// A literal amount only moves bits, so the operand is cut where the discarded bits begin and the
// piece below is moved up. A witnessed one is split into a whole number of 64-bit limbs and the
// rest, which is read out of the powers-of-two table the narrow shifts use, and every limb of the
// operand is split at that rest into two range-checked halves.
//
// `x` is every bit, so each `<<` below discards some, and Noir's `<<` wraps.
fn main(x: u128, n: u128, m: u128) {
    assert(x << 3 == 340282366920938463463374607431768211448);
    assert(x << 127 == 170141183460469231731687303715884105728);
    assert(x << n == 340282366920938462282782986714356908032);
    assert(x >> m == 268435455);
    assert(x >> 3 == 42535295865117307932921825928971026431);

    // An amount a loop computes is pure: known wherever the constraints are built, but not a
    // literal. Nothing about it is witnessed, so each limb's split is by a constant power of two.
    let mut low = 0;
    let mut high = 0;
    for i in 0..4 {
        let amount = (i * 40) as u128;
        low += (x >> amount) & 0xffff;
        high += (x << amount) >> 120;
    }
    assert(low == 196860);
    assert(high == 1020);
}
