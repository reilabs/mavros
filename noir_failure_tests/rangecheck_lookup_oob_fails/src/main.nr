// A value with no row in the rangecheck table it is checked against must *trap* at witness
// generation, not corrupt the VM.
//
// `200 + 100` overflows `u8`, so the addition's 8-bit rangecheck is handed `300` — one past the
// last row of the 256-entry table. `rngchk_field` (and `forward_kv_lookup_emit` beside it) bumps
// the multiplicity of the row a key lands on through raw pointer arithmetic, so before
// `table_row_index` gated that bump, this wrote past the end of `multiplicities_wit` and then died
// indexing the lookup tape in phase 2 ("index out of bounds: the len is 22 but the index is 257").
//
// The exact VM path depends on how `LookupSpilling` sizes this check: if it ever picks the
// decomposition form over the lookup form the program still rejects, but stops exercising the
// bump. `table_row_index`'s unit tests in `vm/src/bytecode.rs` are the precise pin — this test is
// the end-to-end one, and covers the whole-pipeline claim that an unsatisfiable range constraint
// reaches the user as a clean witgen failure.
fn main(x: u8, y: u8, e: pub u8) {
    let z = x + y;
    assert_eq(z, e);
}
