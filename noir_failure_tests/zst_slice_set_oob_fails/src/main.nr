struct P {}

fn main(idx: u32) {
    let mut s: [P] = @[P {}, P {}];
    s[idx] = P {};
    assert(s.len() == 2);
}
