struct P {}

fn main(n: u32) {
    let s: [P] = @[];
    let s = if n == 1 { s.push_back(P {}) } else { s };
    let (s2, _p) = s.pop_back();
    assert(s2.len() == 0);
}
