// GENERATED FILE -- do not edit by hand.
//
// Rendered by `mavros-int-semantics`'s `corpus` module from the model itself. Regenerate with
// `MAVROS_BLESS=1 cargo test -p mavros-int-semantics`, and read the diff: a program whose
// expected answer moved is a program whose *meaning* moved.
//
// This program must be REJECTED. `>>` on signed 64-bit operands `1` and `-9223372036854775808` is
// `Reject::ShiftAmount` in the model, so a run that produces a proof means the guard that owes
// this rejection is missing.
//
// Both operands are witnesses, which is the half the hand-written `noir_failure_tests` do not
// cover: those put the operands behind a function so the *pure* check is under test, whereas
// here the check is owed by the witness lowering instead.
//
// The amount here is *negative* rather than merely too large, and the width is the widest
// signed one rather than the narrowest rejecting one. That pairing is deliberate: it is the
// only shape that reaches `shift_guard`'s negative-amount conjunct, which is dead at every
// narrower width because the widening cast ahead of the bound test clears the sign bit.
//
// The assertion is written against the answer a total backend produces anyway, so removing
// the guard makes this program PASS -- which an expect-failure row reports as a failure.

fn main(a: i64, b: i64) {
    assert_eq(a >> b, 1);
}
