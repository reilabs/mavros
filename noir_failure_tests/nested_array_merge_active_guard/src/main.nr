fn main(outer: bool, inner: bool, x: u64, divisor: u64) {
    // Merging either branch must not suppress an active arm's division error.
    let arr = if outer {
        if inner { [[x / divisor; 2]; 2] } else { [[x; 2]; 2] }
    } else {
        [[7; 2]; 2]
    };
    assert_eq(arr[0][0], x);
}
