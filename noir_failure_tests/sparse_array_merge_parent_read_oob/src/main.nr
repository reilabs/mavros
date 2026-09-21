// The parent read is active even when the nested write and merge arm are not.
fn main(outer: bool, inner: bool) {
    let original: [[Field; 2]; 128] = [[0; 2]; 128];
    for i in 127..129 {
        if outer {
            let row = original[i];
            let result = if inner {
                let mut changed_row = row;
                changed_row[0] = 1;
                let mut changed = original;
                changed[i] = changed_row;
                changed
            } else {
                original
            };
            assert_eq(result[1][0], 0);
        }
    }
}
