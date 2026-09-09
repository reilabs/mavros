fn main(flag: bool) {
    let mut arr: [Field; 3] = [2, 5, 9];
    for i in 0..4 {
        if flag {
            arr[i] = 7;
        }
    }
    assert_eq(arr[0], 7);
}
