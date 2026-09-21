fn main(outer: bool, c: bool) {
    let a0: [Field; 128] = [0; 128];
    for i in 127..129 {
        let mut a = a0;
        if outer {
            // This must fail at i == 128 even when the inner arm is inactive.
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
