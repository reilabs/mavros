// `x` is private and unused: the compiler elides its witness columns, so the
// manifest records it with no column and `z` moves down behind the public `y`.
fn main(x: Field, y: pub Field, z: Field) -> pub Field {
    y + z
}
