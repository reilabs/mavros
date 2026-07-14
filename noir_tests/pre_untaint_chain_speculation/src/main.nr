// PRE speculation at the pre-untaint site (stage 2): the loop body evaluates an invariant
// witness-dependent field-mul chain 10 links deep. Each post-untaint PRE site hoists one link
// per run (a hoisted carrier only unlocks the next link's key at the *next* site), so the eight
// post-untaint sites strand the chain's tail in the loop, paying its rows per unrolled
// iteration. The pre-untaint site hoists one link before untaint, pulling the whole chain out
// within the sites that remain and converting those per-iteration rows into per-entry rows.
fn main(x: Field, y: Field) {
    let mut acc: Field = 0;
    for _i in 0..20 {
        acc += x * y * y * y * y * y * y * y * y * y * y;
    }
    // x = 2, y = 3: 2 * 3^10 = 118098 per iteration, 20 iterations.
    assert(acc == 2361960);
}
