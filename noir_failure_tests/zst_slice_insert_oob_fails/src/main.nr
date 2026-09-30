struct P {}

fn main(idx: u32) {
    let s: [P] = @[P {}, P {}];
    let s2 = s.insert(idx, P {});
    assert(s2.len() == 3);
}
