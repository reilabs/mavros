struct P {}

fn grow(s: &mut [P]) {
    *s = s.push_back(P {});
}

fn main(expected: u32) {
    let mut r: [P] = @[P {}];
    grow(&mut r); // 2
    grow(&mut r); // 3
    assert(r.len() == expected);
}
