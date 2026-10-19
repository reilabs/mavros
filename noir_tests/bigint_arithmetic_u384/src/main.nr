// Exercise every limb, narrowing/widening casts, and arithmetic on cast results.
fn main(x: u64, y: u64) -> pub [u64; 6] {
    let high: u384 = 1 << 382;
    let a = high + (x as u384);
    let b = (y as u384) + 1;
    let product = a * b;
    assert(product / b == a);
    assert(product % b == 0);
    assert(a > b);
    let bits = ((a ^ high) | (b << 368)) & ((1 as u384) << 369 | 255);
    let low = product as u16;
    let mixed = (low as u32) + (x as u32);
    let widened = (mixed as u384) + high;
    [(a + b - high) as u64, (product >> 382) as u64,
     low as u64, (bits >> 368) as u64,
     ((a >> (y as u384)) as u16) as u64, (widened - high) as u64]
}
