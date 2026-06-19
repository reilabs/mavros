// realistic: a loop that mutates the elements of a mutable array and reads them
// back through a second loop. The array is threaded as a loop-carried block
// parameter and indexed by the dynamic loop counter, so the points-to analysis
// must track it across the loop (the array group stays collapsed, not peeled);
// mem2reg promotes the scalar `sum` accumulator. End-to-end correctness guard.
fn main(seed: Field, expected: pub Field) {
    let mut acc = [seed, seed + 1, seed + 2, seed + 3];
    for i in 0..4 {
        acc[i] = acc[i] * 2;
    }
    let mut sum = 0;
    for i in 0..4 {
        sum = sum + acc[i];
    }
    assert_eq(sum, expected);
}
