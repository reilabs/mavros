// 2-D witness-indexed lookup: arr[i][j] with both i and j witness-tainted.
//
// Expected: arr[1][2] = 22. The descriptor arr[i] is a WitnessOf(Array<F, 4>)
// extending step; _[j] then saturates with both keys against the flat 3x4
// table.

fn main(i: u32, j: u32) {
    let arr: [[Field; 4]; 3] = [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ];
    let v = arr[i][j];
    assert_eq(v, 22);
}
