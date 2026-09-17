struct P {}

fn main(idx: u32) {
    let s: [P] = @[];
    let (_s2, _p) = s.remove(idx);
}
