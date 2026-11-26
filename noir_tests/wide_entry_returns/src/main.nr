// Every return wider than one element is published as its limbs: on bn254 all but the `u64`, on
// Goldilocks all of them. Each of those holds a value past its field's modulus, and the `u256`'s
// and the `u64`'s limbs differ, so a limb out of place shows.
fn main(x: u256, y: i300, z: u64) -> pub (u256, [i300; 2], u64, u512) {
    (x + 1, [y, -y], z + 1, 0xff << 504)
}
