struct P {}

fn main(w: Field, expected: u32) {
    let s: [P] = @[];
    let mut l = 5;
    if w == 1 {
        let (s2, _p) = s.pop_back();
        l = s2.len();
    }
    assert(l == expected);
}
