// Witness-indexed lookups on multi-dimensional arrays.
// The outer index is a witness so the resulting inner array is hint-reconstructed
// element-by-element with per-leaf lookups into the flattened original array.

fn main(i: u32, j: u32) {
    // 2D array of Field, witness outer index -- result is an inner array
    let arr_2d: [[Field; 3]; 2] = [
        [10, 20, 30],
        [40, 50, 60],
    ];
    let row = arr_2d[i];
    assert_eq(row[0], 40);
    assert_eq(row[1], 50);
    assert_eq(row[2], 60);

    // Witness outer index followed by witness inner index on the same array
    let cell = arr_2d[i][j];
    assert_eq(cell, 60);

    // 3D array of Field -- two levels of nesting, witness outer index
    let arr_3d: [[[Field; 2]; 2]; 2] = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ];
    let plane = arr_3d[i];
    assert_eq(plane[0][0], 5);
    assert_eq(plane[0][1], 6);
    assert_eq(plane[1][0], 7);
    assert_eq(plane[1][1], 8);

    // 2D array of u32 with witness outer index
    let arr_u32: [[u32; 2]; 3] = [
        [100, 200],
        [300, 400],
        [500, 600],
    ];
    let row_u32 = arr_u32[i];
    assert_eq(row_u32[0], 300);
    assert_eq(row_u32[1], 400);
}
