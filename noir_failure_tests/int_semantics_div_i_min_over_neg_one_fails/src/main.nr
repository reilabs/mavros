// GENERATED FILE -- do not edit by hand.
//
// Rendered by `mavros-int-semantics`'s `corpus` module from the model itself. Regenerate with
// `MAVROS_BLESS=1 cargo test -p mavros-int-semantics`, and read the diff: a program whose
// expected answer moved is a program whose *meaning* moved.
//
// This program must be REJECTED. `/` on signed 8-bit operands `-128` and `-1` is
// `Reject::DivOverflow` in the model, so a run that produces a proof means the guard that owes
// this rejection is missing.
//
// Both operands are witnesses, which is the half the hand-written `noir_failure_tests` do not
// cover: those put the operands behind a function so the *pure* check is under test, whereas
// here the check is owed by the witness lowering instead.
//
// The model declines to specify an answer for this input, so there is nothing to assert
// against. The multiply by a witness zero keeps the result live -- a dead witness operation
// loses its rejection -- without claiming a value the model does not have.

fn main(a: i8, b: i8, zero: i8) {
    let r = a / b;
    assert_eq(r * zero, 0);
}
