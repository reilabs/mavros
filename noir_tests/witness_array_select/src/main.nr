fn main(flag: Field, x: Field, y: Field) {
    // Witness-dependent branch where both sides produce an array
    let arr: [Field; 3] = if flag == 1 {
        [x, x + 1, x + 2]
    } else {
        [y, y + 1, y + 2]
    };

    // Use the merged array so it's not dead code
    assert_eq(arr[0] + arr[1] + arr[2], x * 3 + 3);
}
