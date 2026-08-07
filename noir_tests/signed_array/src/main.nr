fn main(idx: u32, v: i32) {
    let arr: [i32; 4] = [-1, 2, -3, 4];
    assert_eq(arr[0], -1);
    assert_eq(arr[2], -3);

    // Witness index into an array of signed elements.
    assert_eq(arr[idx], 2);

    // Signed values written through a mutable array.
    let mut m: [i32; 3] = [0; 3];
    m[0] = v;
    m[1] = -v;
    m[2] = v * 2;
    assert_eq(m[0] + m[1], 0);
    assert_eq(m[2], -14);
    assert_eq(m[idx], 7);

    // Signed accumulator over a loop.
    let mut sum: i32 = 0;
    for i in 0..4 {
        sum += arr[i];
    }
    assert_eq(sum, 2);

    // Signed minimum via a comparison over the array.
    let mut lo: i32 = arr[0];
    for i in 0..4 {
        if arr[i] < lo {
            lo = arr[i];
        }
    }
    assert_eq(lo, -3);
}
