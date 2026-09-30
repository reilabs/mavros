// `late_folded_dead_shift_fails` at 128 bits, where the check that has to survive belongs to the
// limb-wise shift rather than to the single-cell one.
fn main(a: u128, b: u128) -> pub u128 {
    let s = a << b;
    let mut x = s;
    x - s
}
