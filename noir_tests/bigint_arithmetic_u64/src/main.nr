// Exercise every limb, narrowing/widening casts, and arithmetic on cast results.
fn main(x: u64, y: u64) -> pub [u64; 6] {
    let high: u64 = 1 << 62;
    let a = high + (x as u64);
    let b = (y as u64) + 1;
    let product = a * b;
    assert(product / b == a);
    assert(product % b == 0);
    assert(a > b);
    let bits = ((a ^ high) | (b << 48)) & ((1 as u64) << 49 | 255);
    let low = product as u16;
    let mixed = (low as u32) + (x as u32);
    let widened = (mixed as u64) + high;
    [(a + b - high) as u64, (product >> 62) as u64,
     low as u64, (bits >> 48) as u64,
     ((a >> (y as u64)) as u16) as u64, (widened - high) as u64]
}
