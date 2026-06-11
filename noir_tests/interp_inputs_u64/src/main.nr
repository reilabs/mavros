// Dedicated fixture for the always-on Prover.toml input-bridge test (its own package root so it
// never races another test on the shared `mavros_debug` directory). main computes
// `x * 2 + (p + 1)` natively in u64 = the same value under either build field.
fn main(x: u64) -> pub u64 {
    let big: u64 = 18446744069414584322; // p + 1, a legal u64
    let prod = x * 2;
    prod + big
}
