// Negative fixture: the interpreter must DETECT a false assertion. The real loop sum is 45;
// the assert claims 46. The sum is computed (not const-folded by the frontend), so this
// compiles cleanly and the falsehood is only caught at interpretation - exactly the oracle
// signal we rely on.
fn main() {
    let mut sum: u32 = 0;
    for i in 0..10 {
        sum = sum + i;
    }
    assert(sum == 46); // wrong on purpose: the true sum is 45
}
