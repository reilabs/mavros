struct P {}

fn main(idx: u32) {
    let s: [P] = @[P {}, P {}];
    let (s2, _p) = s.remove(idx);
    assert(s2.len() == 1);
}
