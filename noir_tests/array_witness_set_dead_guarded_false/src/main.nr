// The guard-off world for a *dead* witness-indexed array set, and the half that stops
// `noir_failure_tests/array_witness_set_dead_guarded_true` from passing vacuously: if the bounds
// check DCE leaves behind were emitted unconditionally rather than inside the branch block, this
// program would reject on an index it never uses. Same program as the sibling, guard off.
fn main(w: Field, i: u32, x: Field) {
    let mut a: [Field; 3] = [1, 2, 3];
    if w == 1 {
        a[i] = 9;
    }
    assert(x == 7);
}
