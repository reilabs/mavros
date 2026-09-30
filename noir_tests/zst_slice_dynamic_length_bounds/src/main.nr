struct P {}

// Every bound below is checked at runtime, sits on its last legal value, and passes.
fn main(n: u32, expected: u32) {
    let s: [P] = @[P {}];
    // A witness-dependent length: no bound here can fold to a constant.
    let s = if n == 2 { s.push_back(P {}) } else { s }; // 2
    let s = s.insert(n, P {}); // index == len -> 3
    let (s, _removed) = s.remove(n); // index == len - 1 -> 2
    let _p = s[n - 1]; // index == len - 1
    let (s, _popped) = s.pop_back(); // 1
    let (s, _popped) = s.pop_back(); // len == 1, the last legal pop -> 0
    assert(s.len() == expected);
}
