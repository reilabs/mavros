// Setting past the end of an array at a witness index must trap at witness generation. The
// rebuild scan `gen_witness_array_set` emits only writes at slots it visits, so without the
// bounds constraint an out-of-range index would silently no-op and the program would verify.
fn main(i: u32) {
    let mut a: [Field; 3] = [1, 2, 3];
    a[i] = 9;
    assert_eq(a[0], 1);
}
