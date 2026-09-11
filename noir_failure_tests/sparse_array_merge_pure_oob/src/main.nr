fn main(flag: bool) {
    let mut arr: [Field; 128] = [2; 128];
    for i in 127..129 {
        if flag {
            arr[i] = 7;
        }
    }
    assert_eq(arr[127], 7);
}
