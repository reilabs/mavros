// PRE speculation through the disequality channel (stage 4): inside sum_quotients the
// divisor is a pure, non-constant parameter and the division sits in the body of a
// zero-trip-able loop dominated by the else arm of a d == 0 branch. The branch fact
// discharges the division's totality gate, licensing the hoist to the guarded preheader.
// (An if d != 0 guard lowers to a boolean-not cast the analysis does not yet see through,
// so the equality branch is written with the guarded loop in the else arm.)
fn sum_quotients(base: u32, d: u32) -> u32 {
    let mut acc: u32 = 0;
    if d == 0 {
        acc = 0;
    } else {
        for i in 0..10 {
            acc += base / d + i;
        }
    }
    acc
}

fn main(w: u32) {
    let mut total: u32 = 0;
    for d in 0..5 {
        total += sum_quotients(1000, d);
    }
    assert(total + w == 21011);
}
