// No `return` key in Prover.toml: the circuit computes its return, so nothing is
// checked against it and the run must pass.
fn main(x: u8) -> pub u8 {
    x + 41
}
