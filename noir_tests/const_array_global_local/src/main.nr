// Global + local constant arrays with identical contents, both witness-indexed. They intern to the
// same Constant::Array, so they should resolve to a single shared lookup table (one identity), not
// two colliding ones.
global G: [Field; 3] = [11, 22, 33];

fn main(idx: u32, out: pub Field) {
    let local: [Field; 3] = [11, 22, 33];
    assert_eq(G[idx], local[idx]); // both witness-indexed; identical contents
    assert_eq(out, G[idx]); // out == 22 for idx = 1
}
