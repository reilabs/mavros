// GENERATED FILE -- do not edit by hand.
//
// Rendered by `mavros-int-semantics`'s `corpus` module from the model itself. Regenerate with
// `MAVROS_BLESS=1 cargo test -p mavros-int-semantics`, and read the diff: a program whose
// expected answer moved is a program whose *meaning* moved.
//
// This program must be REJECTED. `<<` on unsigned 8-bit operands `1` and `8` is
// `Reject::ShiftAmount` in the model, so a run that produces a proof means the guard that owes
// this rejection is missing.
//
// Both operands are witnesses, which is the half the hand-written `noir_failure_tests` do not
// cover: those put the operands behind a function so the *pure* check is under test, whereas
// here the check is owed by the witness lowering instead.
//
// The assertion is written against the answer a total backend produces anyway, so removing
// the guard makes this program PASS -- which an expect-failure row reports as a failure.

fn main(a: u8, b: u8) {
    assert_eq(a << b, 1);
}
