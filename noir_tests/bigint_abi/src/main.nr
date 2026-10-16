// The entry-point ABI uses little-endian u128 limbs; arithmetic stays wide internally.
fn main(a: [u128; 2], b: [u128; 2]) -> pub [u128; 2] {
    let lhs = (a[0] as u256) | ((a[1] as u256) << 128);
    let rhs = (b[0] as u256) | ((b[1] as u256) << 128);
    let sum = lhs + rhs;
    [sum as u128, (sum >> 128) as u128]
}
