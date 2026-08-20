// Setting past the end of a slice at a witness index must trap at witness generation. The
// rebuild scan `gen_witness_array_set` emits only writes at slots it visits, so without the
// bounds constraint an out-of-range index would silently no-op and the program would verify.
fn main(i: u32) {
    let mut s: [Field] = [1, 2, 3];
    s[i] = 9;
    assert_eq(s[0], 1);
}
