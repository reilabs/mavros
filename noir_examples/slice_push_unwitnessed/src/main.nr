fn main(a: u8) -> pub u8 {
    let mut s: [u8] = [];
    s = s.push_back(a);
    assert(s.len() == 1);
    s[0]
}
