struct P {}

fn main(w: u8, expected: u32) {
    let w = w as Field;

    let s: [P] = @[];
    let mut l = 5;
    if w == 1 {
        let (s2, _removed) = s.remove(0);
        l = s2.len();
    }
    assert(l == expected);
}
