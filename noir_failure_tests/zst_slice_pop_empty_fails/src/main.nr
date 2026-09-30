struct P {}

fn main(x: u32) {
    let s: [P] = @[];
    let (s2, p) = s.pop_back();
    assert(s2.len() == x);
}
