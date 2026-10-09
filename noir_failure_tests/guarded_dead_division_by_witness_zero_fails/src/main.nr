// A dead division by a witness zero in a taken branch must be rejected.
//
// Each dead-code elimination run before taint inference peels one level, replacing the outermost
// division with its zero check, so seven levels outlast the runs and `a / b` is still present,
// dead, when the branch becomes a guard. A guarded division by a witness has no separate check (its
// lowering is the check), so the cleanup before that lowering must keep it.
fn main(a: u64, b: u64, c: u64, w: bool) {
    if w {
        let _ = a / b / c / c / c / c / c / c;
    }
}
