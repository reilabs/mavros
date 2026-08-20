// Early `return` out of a witness-conditional arm must stay rejected by the frontend. The
// `writes_under_witness_cf` rule in `witness_taint_inference/builder.rs` gives `ArraySet` and the
// slice ops no control-flow taint because every witness arm reconverges at
// `CFG::get_merge_point`; an early return is one of the two shapes that could leave an arm
// without reconverging (`break`/`continue` is the other).
//
// Inputs are supplied deliberately: without them this would keep "failing" on missing prover
// inputs even if the frontend ever started accepting early returns, i.e. it would pass for the
// wrong reason and stop guarding the rule above.
fn main(x: Field, y: pub Field) -> pub Field {
    if x == 3 {
        return x + y;
    }
    let acc = x * y;
    acc + 1
}
