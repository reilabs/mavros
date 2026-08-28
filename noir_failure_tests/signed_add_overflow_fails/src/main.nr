// An unguarded pure signed `+` that leaves the signed range must be rejected.
//
// The signed check is a different predicate from the unsigned one, not a re-parameterisation of it:
// two's-complement addition overflows exactly when the operands share a sign and the result does
// not, which is a question about _sign bits_ rather than about magnitude. The unsigned test would
// answer "no overflow" here -- 120 + 10 is 130, comfortably inside `[0, 256)` -- so this is the
// case that separates them.
//
// Noir rejects it as "attempt to add with overflow" on the same terms; the type decides which
// range the result has to fit, and `i8` stops at 127.

fn add_i8(a: i8, b: i8) -> i8 {
    a + b
}

fn main(w: Field) {
    // Keeps `main` from folding away entirely; the addition below is what is under test.
    assert(w != 0);
    // 120 + 10 is 130, one past `i8`'s 127. `-126` is the wrapped answer.
    assert(add_i8(120, 10) == -126);
}
