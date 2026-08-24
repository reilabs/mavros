// `break`/`continue` out of a witness-conditional arm, in the one place they are legal.
//
// `writes_under_witness_cf` in `witness_taint_inference/builder.rs` gives `ArraySet` and the slice
// ops no control-flow taint, on the grounds that every witness arm reconverges at
// `CFG::get_merge_point` so the two worlds are reconciled by the merge phi. Two shapes could leave
// an arm without reconverging: an early `return`, and `break`/`continue`. The frontend rejects both
// in *constrained* functions (pinned by `noir_failure_tests/{early_return_field,
// break_in_constrained_loop, continue_in_constrained_loop}`), so the only live case is the one this
// test covers — `break`/`continue` in an unconstrained function, where the rule's argument is that
// the jump target is still inside the enclosing loop's region and the arms therefore still join.
//
// The failure tests pin *rejection*, so they would catch a frontend that started accepting these;
// nothing pinned the behaviour where they are already accepted. Both loops write under a
// witness-derived condition and the expectations are public inputs, so the same circuit is checked
// against the arm being taken at different iterations.
unconstrained fn breaking(w: u32) -> [Field; 4] {
    let mut a: [Field; 4] = [0; 4];
    for i in 0..4 {
        if i == w {
            a[0] = 1;
            break;
        }
        a[1] = 2;
    }
    a
}

unconstrained fn continuing(w: u32) -> [Field; 4] {
    let mut a: [Field; 4] = [0; 4];
    for i in 0..4 {
        if i == w {
            continue;
        }
        a[2] = 3;
    }
    a
}

fn main(w: u32, e0: pub Field, e1: pub Field, e2: pub Field) {
    let a = unsafe { breaking(w) };
    assert_eq(a[0], e0);
    assert_eq(a[1], e1);

    let b = unsafe { continuing(w) };
    assert_eq(b[2], e2);
}
