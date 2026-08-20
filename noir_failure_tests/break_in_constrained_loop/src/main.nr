// `break` out of a witness-conditional arm must stay rejected by the frontend. The
// `writes_under_witness_cf` rule in `witness_taint_inference/builder.rs` gives `ArraySet` no
// control-flow taint because every witness arm reconverges at `CFG::get_merge_point`; a
// constrained `break` is one of the two shapes that could leave an arm without reconverging.
fn main(w: u32, e: pub Field) {
    let mut a: [Field; 4] = [0; 4];
    for i in 0..4 {
        if i == w {
            a[0] = 1;
            break;
        }
    }
    assert_eq(a[0], e);
}
