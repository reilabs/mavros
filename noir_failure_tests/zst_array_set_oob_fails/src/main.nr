struct P {}

fn main(idx: u32) {
    let mut a: [P; 2] = [P {}, P {}];
    a[idx] = P {};
}
