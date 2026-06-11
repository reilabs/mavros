// Fixture for the tolerated-error file invariant. `wrapping_sub` lives in the stdlib's
// ops/arith.nr, whose `wrapping_sub_hlp` carries a 2^128 Field constant that does not
// elaborate under Goldilocks, making arith.nr a *tolerated-error* file there. A program that
// REACHES code from such a file must be rejected loudly (the code may be silently mistyped),
// not interpreted/lowered. Under bn254 the file is clean and this program computes 2.
use std::ops::WrappingSub;

fn main(x: u64) -> pub u64 {
    x.wrapping_sub(1)
}
