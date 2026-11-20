// No Prover.toml, but the ABI declares a parameter: input loading must fail
// before witgen runs.
fn main(x: u8) -> pub u8 {
    x + 1
}
