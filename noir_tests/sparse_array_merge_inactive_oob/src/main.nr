fn main(outer: bool, flag: bool, index: u32) {
    check_outer_write(outer, flag);
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

fn check_outer_write(outer: bool, c: bool) {
    let a0: [Field; 128] = [0; 128];
    for i in 127..129 {
        let mut a = a0;
        if outer {
            // An inactive outer branch must not fail at i == 128.
            a[i] = 1;
            let b = if c {
                let mut t = a;
                t[0] = 2;
                t
            } else {
                a0
            };
            assert_eq(b[1], 0);
        }
    }
}
