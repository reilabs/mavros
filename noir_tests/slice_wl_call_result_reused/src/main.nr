// A slice op applied to a call result whose callee signature was lifted, at a call site whose
// own shapes call the result pure.
//
// `extend`'s *joined* return shape is witness-length (because of the guarded call below), so its
// purified signature returns a `(physical, log_len, start)` window at **every** site — including
// the in-branch one, where this site's shapes say the result is still pure-length (the taint only
// lands at the merge). The `t.push_back(3)` that consumes it must therefore be rewritten into its
// window form on the strength of its *operand*, not of its result's shape; keying only on the
// result left a bare `SlicePush` holding a 3-tuple and crashed `Types` with
// "Type is not an array: Tuple<Slice<Field>, u32, u32>".
fn extend(s: [Field], v: Field) -> [Field] {
    s.push_back(v)
}

fn main(w: Field, e_last: pub Field, elen: pub Field) {
    let mut s: [Field] = [1];
    if w == 1 {
        let t = extend(s, 2);
        s = t.push_back(3);
    }
    s = extend(s, 7);
    assert_eq(s.len() as Field, elen);
    assert_eq(s[0], 1);
    let last = s[s.len() - 1];
    assert_eq(last, e_last);
}
