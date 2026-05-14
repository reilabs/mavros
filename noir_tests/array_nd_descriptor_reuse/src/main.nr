// Descriptor reuse: bind a row of a 2-D array to a let, then index it twice
// with different witness indices. Each saturating step should resolve through
// the same root array.

fn main(i: u32, j: u32, k: u32) {
    let arr: [[Field; 4]; 3] = [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ];
    let row = arr[i];
    let a = row[j];
    let b = row[k];
    // arr[1][2] = 22, arr[1][3] = 23 -> sum = 45.
    assert_eq(a + b, 45);
}
