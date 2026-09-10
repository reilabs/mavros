// No `return` key in Prover.toml: guard=0, so the nonzero actual return must
// pass against the zero-filled declared return.
fn main(x: Field) -> pub Field {
    x + 41
}
