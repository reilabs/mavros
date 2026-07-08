fn main(a: Field) -> pub Field {
    let mut s: [Field] = &[];
    s = s.push_back(a);
    assert(s.len() == 1);
    s[0]
}
