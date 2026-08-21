// The out-of-range write must still trap even though nothing reads the result. DCE deletes an
// `ArraySet` whose result is dead, and the bounds check is created later, by the witness-array
// lowering -- so without a dead-op rewrite the deletion takes the only thing that could fail with
// it. Noir keeps parity here by re-inserting the check when it drops an unused array access
// (`ssa/opt/die/array_oob_checks.rs`).
fn main(i: u32, x: Field) {
    let mut a: [Field; 3] = [1, 2, 3];
    a[i] = 9;
    assert(x == 7);
}
