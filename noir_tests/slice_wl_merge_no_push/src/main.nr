// Differing-length merge with no slice ops anywhere: the only witness-length signal is the
// join of two pure slices of different static lengths under a witness condition. The merged
// length must be witness-tainted even though no push/pop ever runs; inputs pick the shorter
// arm so a merge that kept the longer static length would fail the length assert.
fn main(w: u8, e0: pub u8, elen: pub u8) {
    let w = w as Field;
    let e0 = e0 as Field;
    let elen = elen as Field;

    let mut s: [Field] = [10, 20, 30];
    if w == 1 {
        s = [7];
    }
    assert_eq(s.len() as Field, elen);
    assert_eq(s[0], e0);
}
