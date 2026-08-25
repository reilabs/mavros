// The same rejection, on the shift lowering that reaches it by the other route: a *witness*
// left-hand side shifted by a pure amount.
//
// `LowerPureGuards` does not see this one -- it takes pure operands only -- so the check is owed by
// `LowerWitnessBitwiseOps::lower_shift`. That lowering has two halves, and only
// `lower_general_shift` used to emit the check, because it needs the bound to size the amount
// decomposition it builds. `lower_constant_amount_shift`, which is the half taken here and the one
// every shift in the corpus takes, builds no decomposition and so paid nothing: it lowers
// `a >> n` to a division by `1 << n` and hands the raw amount to the backend shift.
//
// That made it the last unchecked shift in the pipeline, and an out-of-range amount there produced
// whatever the backend's own masking happened to give. It was never the right answer, and which
// wrong answer it was moved with the VM: at one point `1u32 << 40` masked to `0`, so the division
// aborted witness generation; once the VM masked the *amount* instead the factor became `256`, the
// division succeeded, and the program returned a wrong value under an unsatisfiable R1CS.
//
// `n` is 40, against a `u32`.

fn main(a: u32) {
    let r = a >> 40;
    assert(r == 0);
}
