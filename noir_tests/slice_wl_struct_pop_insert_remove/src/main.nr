// pop/insert/remove on a *witness-length* slice of multi-leaf struct elements: the two-result
// and shift rewrites must fan out per leaf under the (physical, log_len, start) window: the
// wl x multi-leaf combination the pure-length struct test cannot reach.
struct P {
    a: Field,
    b: Field,
}
fn main(w: Field, e_a: pub Field, e_len: pub Field) {
    let mut s: [P] = [P { a: 1, b: 2 }];
    if w == 1 {
        s = s.push_back(P { a: 3, b: 4 });
    }
    let (s2, last) = s.pop_back();
    assert_eq(last.a, e_a);
    assert_eq(s2.len() as Field, e_len);
    let s3 = s2.insert(0, P { a: 5, b: 6 });
    assert_eq(s3[0].b, 6);
    let (s4, first) = s3.remove(0);
    assert_eq(first.a, 5);
    assert_eq(s4.len() as Field, e_len);
}
