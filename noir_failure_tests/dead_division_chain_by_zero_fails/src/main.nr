// A division by a witness zero must be rejected even when its quotient is never used.
//
// Every division here is dead. Each dead-code elimination run before taint inference replaces the
// outermost one with its zero check but keeps the one below, whose quotient it read, so each run
// peels one level. Seven levels outlast those runs, so `a / b` is dead but still present when the
// witness lowerings begin. An unsigned division by a witness is checked by its lowering, not by a
// separate assertion, so it must survive the cleanup before that lowering, or nothing checks `b`.
fn main(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64) {
    let _ = a / b / c / d / e / f / g / h;
}
