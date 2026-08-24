// Dead *and* guarded: the out-of-range write is inside a witness branch, and nothing ever reads
// the array. Both halves of the rewrite have to hold at once — DCE must leave the bounds check
// behind (`Config::rewrite_dead_seq_access`), and untaint must predicate it on the branch
// condition rather than leaving it unconditional.
//
// `array_witness_set_dead_oob_fails` covers dead-but-unguarded and `array_witness_set_guarded_true`
// covers guarded-but-live; neither reaches the intersection, which is the case the DCE rewrite's
// docblock used to claim was unsupported. It is not: the rewrite only ever runs pre-untaint, where
// the guard is still ordinary control flow, so the `AssertCmp` lands inside the branch block and
// untaint predicates it along with the rest of that block.
//
// Guard on, so the write really happens and the index really is out of range: this must reject.
// `noir_tests/array_witness_set_dead_guarded_false` is the same program with the guard off.
fn main(w: Field, i: u32, x: Field) {
    let mut a: [Field; 3] = [1, 2, 3];
    if w == 1 {
        a[i] = 9;
    }
    assert(x == 7);
}
