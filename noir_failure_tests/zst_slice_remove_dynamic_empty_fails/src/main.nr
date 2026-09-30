struct P {}

fn main(n: u32) {
    let s: [P] = @[];
    // A witness-dependent length: the bound cannot fold to a constant.
    let s = if n == 1 { s.push_back(P {}) } else { s };
    let (s, _removed) = s.remove(0);
    assert(s.len() == 0);
}
