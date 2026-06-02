// Mutating a copy of a constant array must NOT corrupt the original. Constant arrays are
// interned (structurally-equal literals share one id), so `a` and `b` start as the same
// `Constant::Array` value; copy-on-write `ArraySet` (RC-balanced like a runtime MkSeq) must
// give `b[0] = 9` its own buffer. `idx` is a witness so the reads are not const-folded away.

fn main(idx: u32, out: pub Field) {
    let a = [1, 2, 3]; // interned Constant::Array
    let mut b = a;
    b[0] = 9; // must not mutate `a`

    assert_eq(a[idx], 1); // original untouched
    assert_eq(b[idx], 9); // copy updated
    assert_eq(out, a[idx]);
}
