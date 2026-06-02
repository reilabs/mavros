// Exercises constant-array forms beyond a plain literal: a repeated array `[x; N]` and nested
// constant arrays. `idx` is a witness so the reads run on the materialized constant arrays
// rather than being const-folded away.

fn main(idx: u32, out: pub Field) {
    let rep: [Field; 4] = [7; 4]; // Literal::Repeated of a constant -> Constant::Array
    let nested: [[Field; 2]; 2] = [[1, 2], [3, 4]]; // nested Constant::Array

    assert_eq(rep[idx], 7); // idx=1
    assert_eq(nested[1][idx], 4); // nested[1][1] == 4 (witness inner index)
    assert_eq(out, rep[idx]); // out == 7
}
