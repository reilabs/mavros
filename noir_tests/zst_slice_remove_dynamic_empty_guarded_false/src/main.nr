struct P {}

fn main(w: Field, n: u32, expected: u32) {
    let s: [P] = @[];
    let s = if n == 1 { s.push_back(P {}) } else { s };
    let mut l = 5;
    if w == 1 {
        let (s2, _removed) = s.remove(0);
        l = s2.len();
    }
    assert(l == expected);
}
