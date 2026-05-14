// 3-D witness-indexed lookup: arr[i][j][k] with all three witness-tainted.
//
// arr[1][0][2] = 102. The chain creates a 1-key descriptor (from arr[i]),
// extends to 2 keys (from desc[j]), then saturates at the 3rd lookup.

fn main(i: u32, j: u32, k: u32) {
    let arr: [[[Field; 3]; 2]; 2] = [
        [[10, 11, 12], [20, 21, 22]],
        [[100, 101, 102], [110, 111, 112]],
    ];
    let v = arr[i][j][k];
    assert_eq(v, 102);
}
