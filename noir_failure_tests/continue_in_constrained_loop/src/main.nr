// The `continue` twin of `break_in_constrained_loop`: the second shape that could let a
// witness-conditional arm leave without reconverging at `CFG::get_merge_point`, which
// `writes_under_witness_cf` relies on when it gives `ArraySet` no control-flow taint.
fn main(w: u32, e: pub Field) {
    let mut a: [Field; 4] = [0; 4];
    for i in 0..4 {
        if i == w {
            continue;
        }
        a[0] = 1;
    }
    assert_eq(a[0], e);
}
