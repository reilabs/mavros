// `0 / 0` on field elements in a taken branch must be rejected, as Noir rejects any field
// division by zero.
//
// The guarded lowering constrains only `r * b == a` while the branch is taken, which at `0 / 0`
// holds for every `r`: without a zero check the quotient would be pinned by nothing, and a prover
// could return any value here. The honest run answers zero, so it is the check that has to refuse
// it, not the answer.
fn main(a: Field, b: Field, w: bool) -> pub Field {
    let mut r = 0;
    if w {
        r = a / b;
    }
    r
}
