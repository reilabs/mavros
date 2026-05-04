//! Shared layout constants for the WASM VM structs.
//!
//! Two consumers must agree on these byte offsets:
//!   - the test harness in `src/bin/test_runner.rs` that populates the struct
//!     in WASM linear memory before invoking `mavros_main`;
//!   - `src/compiler/llssa_llvm_codegen.rs`, which emits GEP/load/store ops
//!     into `vm_ptr` for the generated forward-pass writes and AD helpers.
//!
//! All offsets are in bytes. wasm32: pointers are 4 bytes, usize/i32 is 4
//! bytes.
//!
//! ## Lookup tables
//!
//! Lookup helpers need per-table state — `multiplicities_wit`-style base,
//! allocated table id, constraint offset, length, etc. — that's only known
//! when a runtime table is first allocated. The VM represents this as
//! `vm.tables: Vec<TableInfo>` plus a small per-kind/per-object cache of the
//! assigned table id. We mirror that here with a host-allocated table-info
//! buffer: slot `i` describes runtime table id `i`. Any per-helper cache
//! state is private to generated WASM, not part of this host ABI.

#![no_std]

/// Size of a pointer in wasm32 linear memory (used to derive layout).
pub const WASM_PTR_SIZE: u32 = 4;

// ── Per-slot table-info record ─────────────────────────────────────────
//
// Field order in `TableInfoSlot` (each field 4 bytes on wasm32):
//   0: mults_base       (ptr; forward multiplicities base)
//   1: inv_cnst_off     (i32; constraints-section start of this table)
//   2: inv_wit_off      (i32; witness-section start of this table)
//   3: num_indices      (i32; matches VM `TableInfo.num_indices`)
//   4: num_values       (i32; matches VM `TableInfo.num_values` — Phase 2
//                        dispatches on this: 0 = width-1, 1 = width-2)
//   5: length           (i32; matches VM `TableInfo.length`)

/// Offsets of fields *within a single `TableInfoSlot`*.
pub const TABLE_INFO_MULTS_BASE_PTR_OFFSET: u32 = 0;
pub const TABLE_INFO_INV_CNST_OFF_OFFSET: u32 = 4;
pub const TABLE_INFO_INV_WIT_OFF_OFFSET: u32 = 8;
pub const TABLE_INFO_NUM_INDICES_OFFSET: u32 = 12;
pub const TABLE_INFO_NUM_VALUES_OFFSET: u32 = 16;
pub const TABLE_INFO_LENGTH_OFFSET: u32 = 20;
/// Total size of one `TableInfoSlot` (6 × 4 = 24 bytes on wasm32).
pub const TABLE_INFO_SLOT_SIZE: u32 = 24;

const _: () =
    assert!(TABLE_INFO_INV_CNST_OFF_OFFSET == TABLE_INFO_MULTS_BASE_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(TABLE_INFO_INV_WIT_OFF_OFFSET == TABLE_INFO_INV_CNST_OFF_OFFSET + 4);
const _: () = assert!(TABLE_INFO_NUM_INDICES_OFFSET == TABLE_INFO_INV_WIT_OFF_OFFSET + 4);
const _: () = assert!(TABLE_INFO_NUM_VALUES_OFFSET == TABLE_INFO_NUM_INDICES_OFFSET + 4);
const _: () = assert!(TABLE_INFO_LENGTH_OFFSET == TABLE_INFO_NUM_VALUES_OFFSET + 4);
const _: () = assert!(TABLE_INFO_SLOT_SIZE == TABLE_INFO_LENGTH_OFFSET + 4);

// ── Forward-pass (witgen) VM struct ─────────────────────────────────────
//
// Cursors (kind-agnostic):
//   - witness, a, b, c (writing cursors into the witness/constraint vectors)
//   - mults_cursor (cursor into the witness multiplicities section; first- use
//     lookups snapshot it, then bump by their table's length)
//   - lookups_a, lookups_b, lookups_c (lookup-tape write cursors)
//   - inputs (immutable input base pointer)
//   - tables_len (count of claimed tables; first-use claim assigns it as the
//     new table id and then bumps it)
//   - tables_cap (capacity of the host-allocated table-info buffer)
//   - tables_ptr (base pointer to the host-allocated table-info buffer)
//   - current_cnst_tables_off (cursor into the tables region of the constraints
//     section)
//   - current_wit_tables_off (cursor into the post-commitment witness tables
//     section, relative to `challenges_start`)
//
pub const WITGEN_WITNESS_PTR_OFFSET: u32 = 0;
pub const WITGEN_A_PTR_OFFSET: u32 = 4;
pub const WITGEN_B_PTR_OFFSET: u32 = 8;
pub const WITGEN_C_PTR_OFFSET: u32 = 12;
pub const WITGEN_MULTS_CURSOR_PTR_OFFSET: u32 = 16;
pub const WITGEN_LOOKUPS_A_PTR_OFFSET: u32 = 20;
pub const WITGEN_LOOKUPS_B_PTR_OFFSET: u32 = 24;
pub const WITGEN_LOOKUPS_C_PTR_OFFSET: u32 = 28;
pub const WITGEN_INPUTS_PTR_OFFSET: u32 = 32;
pub const WITGEN_TABLES_LEN_OFFSET: u32 = 36;
pub const WITGEN_TABLES_CAP_OFFSET: u32 = 40;
/// Base pointer to the table-info buffer. Runtime table id `i` lives at
/// `tables_ptr + i * TABLE_INFO_SLOT_SIZE`.
pub const WITGEN_TABLES_PTR_OFFSET: u32 = 44;
pub const WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET: u32 = 48;
pub const WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET: u32 = 52;
pub const WITGEN_VM_STRUCT_SIZE: u32 = 56;

// ── AD VM struct ────────────────────────────────────────────────────────
//
// Cursors (kind-agnostic):
//   - out_da, out_db, out_dc (immutable output bases)
//   - ad_coeffs (cursor — advances on NextDCoeff)
//   - current_wit_off (next fresh witness index)
//   - ad_coeffs_base (immutable base for random-access reads)
//   - current_lookup_wit_off (cursor for lookup-section witness writes)
//   - current_cnst_tables_off, current_wit_tables_off,
//     current_wit_multiplicities_off (table-region cursors; first-use lookups
//     snapshot and bump)

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
pub const AD_VM_STRUCT_SIZE: u32 = 40;

// Static shape checks — if someone renumbers a field, the build breaks here
// rather than at runtime in WASM.
const _: () = assert!(WITGEN_A_PTR_OFFSET == WITGEN_WITNESS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_B_PTR_OFFSET == WITGEN_A_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_C_PTR_OFFSET == WITGEN_B_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_MULTS_CURSOR_PTR_OFFSET == WITGEN_C_PTR_OFFSET + WASM_PTR_SIZE);
const _: () =
    assert!(WITGEN_LOOKUPS_A_PTR_OFFSET == WITGEN_MULTS_CURSOR_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_LOOKUPS_B_PTR_OFFSET == WITGEN_LOOKUPS_A_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_LOOKUPS_C_PTR_OFFSET == WITGEN_LOOKUPS_B_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_INPUTS_PTR_OFFSET == WITGEN_LOOKUPS_C_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_TABLES_LEN_OFFSET == WITGEN_INPUTS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_TABLES_CAP_OFFSET == WITGEN_TABLES_LEN_OFFSET + 4);
const _: () = assert!(WITGEN_TABLES_PTR_OFFSET == WITGEN_TABLES_CAP_OFFSET + 4);
const _: () =
    assert!(WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET == WITGEN_TABLES_PTR_OFFSET + WASM_PTR_SIZE);
const _: () =
    assert!(WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET == WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET + 4);
const _: () = assert!(WITGEN_VM_STRUCT_SIZE == WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET + 4);
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
const _: () = assert!(AD_VM_STRUCT_SIZE == AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET + 4);
