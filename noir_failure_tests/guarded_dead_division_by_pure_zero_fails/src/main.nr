// `guarded_dead_division_by_witness_zero_fails` with a witness dividend and a pure zero divisor.
//
// One witness operand is enough for a guarded division to be left to its lowering, which rejects
// a pure zero divisor as it does a witnessed one, so this division has to survive to it as well.
fn zero() -> u64 {
    0
}

fn main(a: u64, c: u64, w: bool) {
    if w {
        let _ = a / zero() / c / c / c / c / c / c;
    }
}
