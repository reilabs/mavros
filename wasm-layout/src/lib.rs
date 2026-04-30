//! Shared layout constants for the WASM VM structs.
//!
//! Two consumers must agree on these byte offsets:
//!   - the test harness in `src/bin/test_runner.rs` that populates the struct
//!     in WASM linear memory before invoking `mavros_main`;
//!   - `src/compiler/llssa_llvm_codegen.rs`, which emits GEP/load/store ops
//!     into `vm_ptr` for the generated forward-pass writes and AD helpers.
//!
//! All offsets are in bytes. wasm32: pointers are 4 bytes, usize/i32 is 4 bytes.

#![no_std]

/// Size of a pointer in wasm32 linear memory (used to derive layout).
pub const WASM_PTR_SIZE: u32 = 4;

// ── Forward-pass (witgen) VM struct ─────────────────────────────────────
// Layout (in order):
//   0: witness cursor                (advances on each __write_witness)
//   1: a cursor                      (advances on each __write_a)
//   2: b cursor                      (advances on each __write_b)
//   3: c cursor                      (advances on each __write_c)
//   4: mults_cursor                  (cursor into the witness multiplicities
//                                     section; first-use lookups snapshot it
//                                     into their per-table slot and then bump
//                                     by the table's length. Mirrors the VM's
//                                     `multiplicities_witness` pointer.)
//   5: lookup-tape a cursor          (one Field per Lookup call)
//   6: lookup-tape b cursor          (one Field per Lookup call)
//   7: lookup-tape c cursor          (one Field per Lookup call)
//   8: input base pointer            (immutable pointer to flattened ABI inputs)
//   9: next_table_idx (i32)          (cursor producing the next free table
//                                     index. Mirrors `vm.tables.len()`.)
//  10: current_cnst_tables_off (i32) (cursor into the constraints table
//                                     section; first-use lookups snapshot and
//                                     bump it by their constraint footprint.)
//  11: current_wit_tables_off (i32)  (cursor into the post-commitment witness
//                                     table section, relative to
//                                     `challenges_start`; first-use lookups
//                                     snapshot and bump it by their witness
//                                     table footprint.)
//  12: rngchk_8_cnst_off (i32)       (sentinel `u32::MAX` until first
//                                     rangecheck-8 Lookup; then snapshots its
//                                     constraints table offset.)
//  13: rngchk_8_wit_off (i32)        (sentinel `u32::MAX` until first
//                                     rangecheck-8 Lookup; then snapshots its
//                                     post-commitment witness table offset.)
//  14: rngchk_8_mults_base (ptr)     (sentinel `null`/`0` until first
//                                     rangecheck-8 Lookup; then snapshots
//                                     `mults_cursor` so subsequent calls
//                                     index into the right slab. Mirrors
//                                     `TableInfo.multiplicities_wit`.)
//  15: rngchk_8_table_idx (i32)      (sentinel `u32::MAX` until first
//                                     rangecheck-8 Lookup; then snapshots
//                                     `next_table_idx`. Mirrors `vm.rgchk_8`.)

pub const WITGEN_WITNESS_PTR_OFFSET: u32 = 0;
pub const WITGEN_A_PTR_OFFSET: u32 = 4;
pub const WITGEN_B_PTR_OFFSET: u32 = 8;
pub const WITGEN_C_PTR_OFFSET: u32 = 12;
pub const WITGEN_MULTS_CURSOR_PTR_OFFSET: u32 = 16;
pub const WITGEN_LOOKUPS_A_PTR_OFFSET: u32 = 20;
pub const WITGEN_LOOKUPS_B_PTR_OFFSET: u32 = 24;
pub const WITGEN_LOOKUPS_C_PTR_OFFSET: u32 = 28;
pub const WITGEN_INPUTS_PTR_OFFSET: u32 = 32;
pub const WITGEN_NEXT_TABLE_IDX_OFFSET: u32 = 36;
pub const WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET: u32 = 40;
pub const WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET: u32 = 44;
pub const WITGEN_RNGCHK_8_CNST_OFF_OFFSET: u32 = 48;
pub const WITGEN_RNGCHK_8_WIT_OFF_OFFSET: u32 = 52;
pub const WITGEN_RNGCHK_8_MULTS_BASE_PTR_OFFSET: u32 = 56;
pub const WITGEN_RNGCHK_8_TABLE_IDX_OFFSET: u32 = 60;
pub const WITGEN_VM_STRUCT_SIZE: u32 = 64;

/// Sentinel for `WITGEN_RNGCHK_8_TABLE_IDX_OFFSET` meaning "rangecheck-8 table
/// not yet allocated by the forward path". Mirrors `Option::None` for
/// `vm.rgchk_8`. The `mults_base` snapshot is checked separately as null,
/// since pointer 0 can never be a real wasm32 multiplicities slot (the host
/// places buffers above `__data_end`).
pub const WITGEN_RNGCHK_8_UNALLOCATED: u32 = u32::MAX;

// ── AD VM struct ────────────────────────────────────────────────────────
// Layout:
//   0:  out_da                  (ptr, fixed base — index into at runtime)
//   1:  out_db                  (ptr, fixed base)
//   2:  out_dc                  (ptr, fixed base)
//   3:  ad_coeffs               (cursor — advances on NextDCoeff)
//   4:  current_wit_off (i32)   (next fresh witness index)
//   5:  ad_coeffs_base           (immutable base for random-access reads)
//   6:  current_lookup_wit_off (i32)
//                                (next absolute witness index in the lookups
//                                 section; bumped per rangecheck DLookup)
//   7:  current_cnst_tables_off (i32)
//                                (cursor into the constraints tables section;
//                                 bumped by each first-use lookup that claims
//                                 a fresh table region. Mirrors the VM's
//                                 `current_cnst_tables_off`.)
//   8:  current_wit_tables_off (i32)
//                                (cursor into the witness tables section;
//                                 bumped by each first-use lookup. Mirrors the
//                                 VM's `current_wit_tables_off`.)
//   9:  current_wit_multiplicities_off (i32)
//                                (cursor into the witness multiplicities
//                                 section; bumped by each first-use lookup.
//                                 Mirrors the VM's
//                                 `current_wit_multiplicities_off`.)
//  10:  rngchk_8_inv_cnst_off (i32)
//                                (sentinel `u32::MAX` until the first
//                                 rangecheck-8 DLookup; then holds the
//                                 absolute `inv_cnst_off` snapshot for the
//                                 rangecheck-8 table — analogous to the VM's
//                                 `vm.rgchk_8: Option<usize>`.)

pub const AD_OUT_DA_PTR_OFFSET: u32 = 0;
pub const AD_OUT_DB_PTR_OFFSET: u32 = 4;
pub const AD_OUT_DC_PTR_OFFSET: u32 = 8;
pub const AD_COEFFS_PTR_OFFSET: u32 = 12;
pub const AD_CURRENT_WIT_OFF_OFFSET: u32 = 16;
pub const AD_COEFFS_BASE_PTR_OFFSET: u32 = 20;
pub const AD_CURRENT_LOOKUP_WIT_OFF_OFFSET: u32 = 24;
pub const AD_CURRENT_CNST_TABLES_OFF_OFFSET: u32 = 28;
pub const AD_CURRENT_WIT_TABLES_OFF_OFFSET: u32 = 32;
pub const AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET: u32 = 36;
pub const AD_RNGCHK_8_INV_CNST_OFF_OFFSET: u32 = 40;
pub const AD_VM_STRUCT_SIZE: u32 = 44;

/// Sentinel for `AD_RNGCHK_8_INV_CNST_OFF_OFFSET` meaning "rangecheck-8 table
/// not yet allocated by the AD path". Mirrors `Option::None` for the VM's
/// `rgchk_8` field. Picked to match `u32::MAX`, which can never be a real
/// constraint offset for any reasonable program size.
pub const AD_RNGCHK_8_UNALLOCATED: u32 = u32::MAX;

// Static shape checks — if someone renumbers a field, the build breaks here
// rather than at runtime in WASM.
const _: () = assert!(WITGEN_VM_STRUCT_SIZE == 10 * WASM_PTR_SIZE + 6 * 4);
const _: () = assert!(AD_VM_STRUCT_SIZE == 6 * WASM_PTR_SIZE + 5 * 4);
const _: () = assert!(WITGEN_A_PTR_OFFSET == WITGEN_WITNESS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_B_PTR_OFFSET == WITGEN_A_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_C_PTR_OFFSET == WITGEN_B_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_MULTS_CURSOR_PTR_OFFSET == WITGEN_C_PTR_OFFSET + WASM_PTR_SIZE);
const _: () =
    assert!(WITGEN_LOOKUPS_A_PTR_OFFSET == WITGEN_MULTS_CURSOR_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_LOOKUPS_B_PTR_OFFSET == WITGEN_LOOKUPS_A_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_LOOKUPS_C_PTR_OFFSET == WITGEN_LOOKUPS_B_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_INPUTS_PTR_OFFSET == WITGEN_LOOKUPS_C_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_NEXT_TABLE_IDX_OFFSET == WITGEN_INPUTS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET == WITGEN_NEXT_TABLE_IDX_OFFSET + 4);
const _: () =
    assert!(WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET == WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET + 4);
const _: () = assert!(WITGEN_RNGCHK_8_CNST_OFF_OFFSET == WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET + 4);
const _: () = assert!(WITGEN_RNGCHK_8_WIT_OFF_OFFSET == WITGEN_RNGCHK_8_CNST_OFF_OFFSET + 4);
const _: () = assert!(WITGEN_RNGCHK_8_MULTS_BASE_PTR_OFFSET == WITGEN_RNGCHK_8_WIT_OFF_OFFSET + 4);
const _: () = assert!(
    WITGEN_RNGCHK_8_TABLE_IDX_OFFSET == WITGEN_RNGCHK_8_MULTS_BASE_PTR_OFFSET + WASM_PTR_SIZE
);
const _: () = assert!(AD_OUT_DB_PTR_OFFSET == AD_OUT_DA_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_OUT_DC_PTR_OFFSET == AD_OUT_DB_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_COEFFS_PTR_OFFSET == AD_OUT_DC_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_CURRENT_WIT_OFF_OFFSET == AD_COEFFS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_COEFFS_BASE_PTR_OFFSET == AD_CURRENT_WIT_OFF_OFFSET + WASM_PTR_SIZE);
const _: () =
    assert!(AD_CURRENT_LOOKUP_WIT_OFF_OFFSET == AD_COEFFS_BASE_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_CURRENT_CNST_TABLES_OFF_OFFSET == AD_CURRENT_LOOKUP_WIT_OFF_OFFSET + 4);
const _: () = assert!(AD_CURRENT_WIT_TABLES_OFF_OFFSET == AD_CURRENT_CNST_TABLES_OFF_OFFSET + 4);
const _: () =
    assert!(AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET == AD_CURRENT_WIT_TABLES_OFF_OFFSET + 4);
const _: () =
    assert!(AD_RNGCHK_8_INV_CNST_OFF_OFFSET == AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET + 4);
