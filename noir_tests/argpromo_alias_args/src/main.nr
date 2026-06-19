// argument promotion guard (C4): the same slot reaches both `&mut` arguments of
// a single call, so the arguments may-alias and promotion must be declined. The
// result must still be correct.
//
// NOTE: contingent on the frontend accepting two `&mut` to one local in one
// call; if it rejects this, the C4 path stays covered by the in-tree unit tests
// and ref_return_alias.
fn add_into(dst: &mut Field, src: &mut Field) {
    *dst = *dst + *src;
}

fn main(x: Field, expected: pub Field) {
    let mut v = x;
    add_into(&mut v, &mut v);
    assert_eq(v, expected);
}
