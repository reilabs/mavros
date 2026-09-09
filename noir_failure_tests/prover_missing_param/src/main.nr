// Prover.toml omits `y`: input loading must fail before witgen runs.
fn main(x: Field, y: Field) -> pub Field {
    x + y
}
