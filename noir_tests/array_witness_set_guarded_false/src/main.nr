// The guard-off world for a witness-indexed array set: the write never happens, so its bounds
// check must be suppressed even though the index is far out of range. `gen_witness_array_set`
// emits the check under the op's guard for exactly this, and the array arm's encoding
// (`sum(hit) == 1`) has to inherit that -- with the guard off no slot is hit, so the sum is zero
// and an unguarded assert would reject a program that must verify.
fn main(w: Field, i: u32, expected0: pub Field) {
    let mut a: [Field; 3] = [1, 2, 3];
    if w == 1 {
        a[i] = 9;
    }
    assert_eq(a[0], expected0);
}
