// On bn254 `x`, `y` and `p.1` cross the entry point as limbs, each with values past the modulus;
// on Goldilocks every integer but `p.0` does.
fn main(x: u256, y: [i300; 2], z: pub u128, p: (u8, u512)) -> pub u32 {
    assert(y[0] == -y[1]);
    assert(x + (z as u256) == 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff);
    ((x >> 248) as u32) + ((p.1 >> 504) as u32) + (p.0 as u32) + ((y[1] >> 290) as u32)
}
