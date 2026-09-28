// `guarded_dead_division_by_witness_zero_fails` on field elements.
//
// A guarded field division has no zero check in its lowering, so it is given one beside it as an
// assertion under the same guard, which no cleanup deletes however dead the division is.
fn main(a: Field, b: Field, c: Field, w: bool) {
    if w {
        let _ = a / b / c / c / c / c / c / c;
    }
}
