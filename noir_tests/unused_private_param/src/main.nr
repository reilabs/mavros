// `x` is private and unused, but its range check reads it, so it keeps a witness
// column: an elided entry parameter of nonzero width comes only from hand-built SSA
// (`compiler/tests/entry_elision.rs`).
fn main(x: u8, y: pub u8, z: u8) -> pub u8 {
    y + z
}
