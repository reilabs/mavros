fn main(a: Field, flag: bool) -> pub Field {
    let mut s: [Field] = &[];
    if flag {
        s = s.push_back(a);
    }
    assert(s.len() == 1);
    s[0]
}
