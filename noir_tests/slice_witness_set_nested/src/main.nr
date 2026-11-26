// Witness-index set of a nested element into a witness-length slice: exercises the leaf-wise
// select recursion in gen_witness_array_set (array-typed elements rebuilt statically, scalar
// leaves selected against the witness index).
fn main(w: u32, e00: pub u8, e01: pub u8, e10: pub u8, e11: pub u8) {
    let e00 = e00 as Field;
    let e01 = e01 as Field;
    let e10 = e10 as Field;
    let e11 = e11 as Field;

    let mut s: [[Field; 2]] = [[1, 2]];
    if w == 1 {
        s = s.push_back([3, 4]);
    }
    s[w] = [5, 6];
    assert_eq(s[0][0], e00);
    assert_eq(s[0][1], e01);
    assert_eq(s[1][0], e10);
    assert_eq(s[1][1], e11);
}
