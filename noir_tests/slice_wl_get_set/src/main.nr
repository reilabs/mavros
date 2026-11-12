// ArrayGet/ArraySet on a witness-length slice with both index kinds: pure-index get and set
// must fold start into the physical index; witness-index get and set must scan with selects
// against the (start, log_len) window.
fn main(w: u32, e0: pub u8, e1: pub u8, e2: pub u8, elen: pub u8) {
    let e0 = e0 as Field;
    let e1 = e1 as Field;
    let e2 = e2 as Field;
    let elen = elen as Field;

    let mut s: [Field] = [10, 20];
    if w == 1 {
        s = s.push_back(30);
    }
    s[0] = 100;
    let a = s[1];
    s[w] = a + 1;
    let b = s[w];
    assert_eq(b, a + 1);
    assert_eq(s[0], e0);
    assert_eq(s[1], e1);
    assert_eq(s[2], e2);
    assert_eq(s.len() as Field, elen);
}
