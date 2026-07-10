// PRE speculation (stage 4): the invariant witness product x * y is evaluated only in the
// body of a zero-trip-able loop, so it is not anticipated at the header. Field
// multiplication is total, which licenses hoisting it onto the entry edge; the loop body
// then reuses one evaluation instead of recomputing it every iteration.
fn main(x: Field, y: Field) {
    let mut acc: Field = 0;
    for _i in 0..50 {
        acc += x * y;
    }
    assert(acc == 750);
}
