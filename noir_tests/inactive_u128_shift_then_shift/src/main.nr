// A guarded `u128` shift by a pure amount past the width, taken with the guard inactive, feeding a
// left shift by a literal.
//
// `k` is pure and always 200, but not a literal, so the limb-wise shift takes it as the amount it
// is on every path: its comparison against the width is asserted only where `c` holds, and the
// split and the barrel are built from it whether or not `c` does. Where `c` is false that is an
// amount past the width, so every constraint they build has to admit one, and `r << 100` then cuts
// whatever `r` is there.
fn main(x: u128, c: bool) -> pub u128 {
    let mut out = 0;
    for i in 0..2 {
        let k = ((i / 1000) + 200) as u128;
        if c {
            let r = x >> k;
            let s = r << 100;
            out += s & 0xffff;
        }
    }
    out
}
