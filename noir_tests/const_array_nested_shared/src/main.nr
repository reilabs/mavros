// Nested / shared constant sub-arrays. The inner array [11, 22, 33] appears three times: twice inside
// G1 and once inside G2. They intern to the same Constant::Array, so after structural dedup in
// constant materialization they must resolve to a SINGLE shared identity (one ArrayData / one lookup
// table), matching the VM's ConstPool. Each inner array below is read by a witness index, so each
// distinct identity would otherwise register its own duplicate lookup table.
global G1: [[Field; 3]; 2] = [[11, 22, 33], [11, 22, 33]]; // two structurally identical inner arrays
global G2: [[Field; 3]; 2] = [[44, 55, 66], [11, 22, 33]]; // shares inner [11, 22, 33] with G1

fn main(i: u32, out: pub Field) {
    // Inner arrays witness-indexed (i is a witness) -> each inner identity registers a lookup table.
    assert_eq(G1[0][i], G1[1][i]); // identical inner arrays within one constant
    assert_eq(G1[0][i], G2[1][i]); // inner array shared across two distinct constants
    assert_eq(out, G1[0][i]); // out == 22 for i = 1
}
