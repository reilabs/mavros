fn main(outer: bool, flag: bool, index: u32) {
    let mut a: [Field; 3] = [1, 2, 3];
    if outer {
        if flag { a[index] = 7; }
    }
    assert_eq(a[1], 2);

    // Keep replay profitable, and make the invalid index pure. The enclosing
    // predicate is false even though the inner branch condition is true.
    let initial = index as Field;
    let mut b: [Field; 128] = [initial; 128];
    for i in 127..129 {
        if outer {
            if flag { b[i] = 7; }
        }
    }
    assert_eq(b[0], initial);
    assert_eq(b[127], initial);
}
