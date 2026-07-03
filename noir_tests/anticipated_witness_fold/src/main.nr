// Exercises the witness-typed anticipated `Cmp{Eq}` fold in SCS (the keep-compatible recast).
//
// The witnessed re-test must reach the post-untaint SCS runs with a bound-to-run establishing
// assert over the *same* value pair, which no directly-written shape achieves: the pre-untaint
// scalar anticipated fold consumes any pair visible before witness typing. The duplicated
// `a + b` defeats that ordering: the conditional-fact channel tracks raw value pairs, so the
// assert over `y` says nothing about the comparison over `x` while they are distinct values,
// and the first CSE run of the pipeline sits post-untaint (witness_spilling), after `eq` has
// become a witnessed comparison feeding a select. Once CSE unifies `y` into `x`, the next SCS
// folds `eq` through the anticipated channel: the select condition is redirected to a hoisted
// `cast true to WitnessOf`, the kept comparison loses its last use, and the equality gadget
// dies instead of lowering to constraint rows.
fn main(a: Field, b: pub Field, c: pub Field) {
    let x = a + b;
    let y = a + b;
    let eq = x == c;
    let r = if eq { c } else { c + 1 };
    assert(y == c);
    assert_eq(r, c);
}
