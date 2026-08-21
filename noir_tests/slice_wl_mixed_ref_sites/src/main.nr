// A `&mut [Field]` helper called from a witness-length site *and* a pure-length one.
//
// `grow`'s pointee is witness-length at the guarded site, so its *joined* shapes purify the
// parameter into a `Ref<(physical, log_len, start)>` — at **every** site, including the
// unconditional one whose own shapes leave `a`'s cell pure-length. Unlike a by-value argument
// there is nothing to materialize at the boundary: the callee writes through the pointer, so a
// copy-out shim would have to turn a window back into a witness-length slice. `close_boundaries`
// widens the caller's cell instead, which is free — a pure-length slice held as a window is just
// `(physical, physical.len(), 0)`.
//
// Before that closure existed the two sides disagreed on the pointee type and `ElideTuples` split
// only the callee's parameter, so the call arrived with the wrong arity and crashed `Types` with a
// bare "Error running opcode Call { .. }".
fn grow(r: &mut [Field], v: Field) {
    *r = (*r).push_back(v);
}

fn main(w: Field, elen_a: pub Field, elen_b: pub Field) {
    // Pure-length site: `a`'s own shapes never make this cell witness-length.
    let mut a: [Field] = [10];
    grow(&mut a, 15);

    // Witness-length site: this is what lifts `grow`'s joined parameter.
    let mut b: [Field] = [20];
    if w == 1 {
        grow(&mut b, 25);
    }

    assert_eq(a.len() as Field, elen_a);
    assert_eq(a[0], 10);
    assert_eq(a[1], 15);
    assert_eq(b.len() as Field, elen_b);
    assert_eq(b[0], 20);
}
