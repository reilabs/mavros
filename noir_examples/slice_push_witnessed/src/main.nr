fn main(a: u8, flag: bool) -> pub u8 {
    let mut s: [u8] = [];
    if flag {
        s = s.push_back(a);
    }
    assert(s.len() == 1);
    s[0]
}
