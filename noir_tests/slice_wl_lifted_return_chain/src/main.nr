// A lifted callee's window returned back out through a caller whose own return slot is pure.
//
// `ext` is lifted by the `ext(b)` site below (its argument is witness-length there), so it returns
// a `(physical, log_len, start)` window at **every** site. `mid` calls it at a site whose own
// shapes still call the result pure-length, so `mid` holds a window that its own joined return
// shape does not know about. `is_window_op` carries that skew *forward* through the ops consuming
// the window, but a return slot is not an op: without `close_boundaries` raising `mid`'s slot too,
// the window reached `Return` and hit the flat refusal
// "ICE: purify_witness_slices: wl slice tuple flows into a pure".
//
// The same closure covers the mirror case — a window flowing into a *callee's* parameter that the
// callee's own joined shapes left pure — which `mid`'s tail call would trip if the lift only ever
// travelled downwards.
fn ext(s: [Field]) -> [Field] {
    s.push_back(9)
}

fn mid(s: [Field]) -> [Field] {
    ext(s)
}

fn main(w: Field, la: pub Field, lc: pub Field) {
    // Pure-length site: `mid`'s own return shape stays pure here.
    let a: [Field] = [1];
    let a2 = mid(a);

    let mut b: [Field] = [1];
    if w == 1 {
        b = b.push_back(2);
    }
    // Witness-length argument: this is what lifts `ext`'s joined parameter and return.
    let c = ext(b);

    assert_eq(a2.len() as Field, la);
    assert_eq(a2[0], 1);
    assert_eq(a2[1], 9);
    assert_eq(c.len() as Field, lc);
    assert_eq(c[0], 1);
}
