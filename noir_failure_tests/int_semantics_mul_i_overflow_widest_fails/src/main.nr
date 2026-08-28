// GENERATED FILE -- do not edit by hand.
//
// Rendered by `mavros-int-semantics`'s `corpus` module from the model itself. Regenerate with
// `MAVROS_BLESS=1 cargo test -p mavros-int-semantics`, and read the diff: a program whose
// expected answer moved is a program whose *meaning* moved.
//
// This program must be REJECTED. `*` on signed 64-bit operands `2` and `6148914691236517205` is
// `Reject::Overflow` in the model, so a run that produces a proof means the guard that owes
// this rejection is missing.
//
// Both operands are witnesses, which is the half the hand-written `noir_failure_tests` do not
// cover: those put the operands behind a function so the *pure* check is under test, whereas
// here the check is owed by the witness lowering instead.
//
// The width here is the *widest* this cell rejects at rather than the narrowest, and there
// is a sibling program at the narrowest. The pair exists because this rejection's bound is
// one the width decides -- a magnitude limit, the amount bound itself, or `INT_MIN` -- so a
// check that is right at eight bits carries no evidence that it is right here.
//
// The assertion is written against the answer a total backend produces anyway, so removing
// the guard makes this program PASS -- which an expect-failure row reports as a failure.

fn main(a: i64, b: i64) {
    assert_eq(a * b, -6148914691236517206);
}
