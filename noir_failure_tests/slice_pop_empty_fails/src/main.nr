// Popping a statically-empty slice must reject: either R1CS generation refuses the program as
// unsatisfiable or witness generation traps; running cleanly is a soundness bug.
fn main() {
    let s: [Field] = [];
    let (_s2, e) = s.pop_back();
    assert_eq(e, 0);
}
