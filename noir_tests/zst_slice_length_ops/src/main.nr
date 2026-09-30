struct P {}

fn main(idx: u32, expected: u32) {
    let s: [P] = @[P {}, P {}, P {}];
    let s = s.push_back(P {}); // 4
    let s = s.push_front(P {}); // 5
    let (s, _popped) = s.pop_back(); // 4
    let (_popped_front, s) = s.pop_front(); // 3
    let s = s.insert(idx, P {}); // 4
    let (s, _removed) = s.remove(idx); // 3
    // A witness-dependent length.
    let s = if idx == 2 { s } else { s.push_back(P {}) }; // 3 when idx == 2

    let a: [P; 2] = [P {}, P {}];
    let v = a.as_vector(); // 2

    assert(s.len() + v.len() == expected);
}
