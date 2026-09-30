struct P {}

fn main(idx: u32) {
    let s: [P] = @[P {}, P {}];
    let _p = s[idx];
}
