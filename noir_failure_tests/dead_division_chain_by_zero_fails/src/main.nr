// A division by a witness zero must be rejected even where its quotient is never used.
//
// Every division here is dead. Each dead-code elimination run before taint inference replaces the
// outermost of them with its zero check but has to keep the one below it, whose quotient that
// division read, so a run peels one level. Seven levels outlast those runs, leaving `a / b` dead
// but present when the witness lowerings begin. An unsigned division by a witness is checked by
// its own lowering rather than by an assertion beside it, so it has to survive the cleanup that
// runs before that lowering, or nothing checks `b` at all.
fn main(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64) {
    let _ = a / b / c / d / e / f / g / h;
}
