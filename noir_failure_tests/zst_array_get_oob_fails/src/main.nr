struct P {}

fn main(idx: u32) {
    let a: [P; 2] = [P {}, P {}];
    let _p = a[idx];
}
