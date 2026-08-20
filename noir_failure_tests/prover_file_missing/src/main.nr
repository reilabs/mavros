// No Prover.toml, but the ABI declares a parameter: input loading must fail
// before witgen runs.
fn main(x: Field) -> pub Field {
    x + 1
}
