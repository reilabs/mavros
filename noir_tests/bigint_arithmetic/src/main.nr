// The literal exceeds both u128 and the BN254 modulus.
fn main(x: u64, y: u64) -> pub [u64; 5] {
    let high: u512 = 0x1000000000000000000000000000000000000000000000000000000000000000000000000000;
    let a = high + (x as u512);
    let b = (y as u512) + 1;
    let product = a * b;
    assert(product / b == a);
    assert(product % b == 0);
    assert(a > b);
    let bits = ((a ^ high) | (b << 200)) & ((1 as u512) << 201 | 255);
    [(a + b - high) as u64, (product >> 300) as u64, product as u64,
     (bits >> 200) as u64, (a >> (y as u512)) as u64]
}
