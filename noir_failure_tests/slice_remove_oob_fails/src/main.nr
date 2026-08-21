// Removing at an out-of-bounds witness index must trap at witness generation.
fn main(i: u32) {
    let s: [Field] = [10];
    let (_s2, e) = s.remove(i);
    assert_eq(e, 10);
}
