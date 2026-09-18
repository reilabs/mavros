fn main(outer: bool, flag: bool, index: u32) {
    let mut a: [Field; 3] = [1, 2, 3];
    if outer {
        if flag { a[index] = 7; }
    }
    assert_eq(a[1], 2);
}
