// realistic: a scalar `&mut` helper called inside a fixed loop. Every unrolled
// call site passes the same clean local, so the parameter promotes (in + out),
// and mem2reg then threads the accumulator slot across the unrolled body.
fn step(acc: &mut Field, x: Field) {
    *acc = *acc * 2 + x;
}

fn main(x: Field, expected: pub Field) {
    let mut acc = 0;
    for _ in 0..5 {
        step(&mut acc, x);
    }
    assert_eq(acc, expected);
}
