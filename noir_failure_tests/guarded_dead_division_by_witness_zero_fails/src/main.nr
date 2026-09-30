// A dead division by a witness zero in a taken branch must be rejected.
//
// Each dead-code elimination run before taint inference replaces the outermost of these divisions
// with its zero check but keeps the one below it, whose quotient that division read, so a run
// peels one level. Seven levels outlast those runs, leaving `a / b` dead but present when the
// branch becomes a guard. A guarded division with a witness operand has no check built beside it:
// its lowering is the check, so the cleanup that runs before that lowering has to keep it.
fn main(a: u64, b: u64, c: u64, w: bool) {
    if w {
        let _ = a / b / c / c / c / c / c / c;
    }
}
