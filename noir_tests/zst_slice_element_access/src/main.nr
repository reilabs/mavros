struct P {}
    
fn main(idx: u32) {
    let mut w: [P] = @[P {}, P {}, P {}];
    let _elem = w[idx]; // in bounds
    w[idx] = P {}; // in bounds
    assert(w.len() == 3);
}
