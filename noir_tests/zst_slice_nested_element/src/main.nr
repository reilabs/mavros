struct P {}

struct Q {
    p: P,
    ps: [P; 2],
}

fn main(idx: u32) {
    let q = Q { p: P {}, ps: [P {}, P {}] };
    let d: [[Q; 2]] = @[[q, q], [q, q]];
    let d = d.push_back([q, q]); // 3
    let _inner: [Q; 2] = d[idx]; // in bounds
    let (d, _last) = d.pop_back(); // 2
    assert(d.len() == 2);
}
