//! Shared layout constants for the WASM VM structs.
//!
//! Two consumers must agree on these byte offsets:
//!   - the test harness in `src/bin/test_runner.rs` that populates the struct
//!     in WASM linear memory before invoking `mavros_main`;
//!   - `src/compiler/llssa_llvm_codegen.rs`, which emits GEP/load/store ops
//!     into `vm_ptr` for the generated forward-pass writes and AD helpers.
//!
//! All offsets are in bytes. wasm32: pointers are 4 bytes, usize/i32 is 4 bytes.
//!
//! ## Lookup tables
//!
//! Lookup helpers (forward `__rngchk_8`, AD `__drngchk_8_ad_call`, future
//! array/spread variants) need per-table state — `multiplicities_wit`-style
//! base, allocated table id, AD constraint offset, length, etc. — that's
//! only known on first use of that lookup *kind*. The VM does this with
//! `vm.tables: Vec<TableInfo>` plus an `Option<usize>` per kind that caches
//! the index assigned at first use. We mirror it here with a fixed-capacity
//! registry: each lookup kind is assigned a compile-time slot index, and
//! that slot's data lives in an inline array of `TableInfoSlot` records
//! inside the VM struct — same shape as the VM's `Vec<TableInfo>`, just
//! addressed by the kind's compile-time slot index. Slots are
//! zero-initialized, so a slot is "unclaimed" iff its `occupancy` field is
//! zero.
//!
//! `MAX_TABLE_KINDS` is the compile-time bound on the number of distinct
//! lookup kinds in any program. Bump if the compiler ever needs more (it
//! will assert at SSA-lowering time).

#![no_std]

/// Size of a pointer in wasm32 linear memory (used to derive layout).
pub const WASM_PTR_SIZE: u32 = 4;

/// Maximum number of distinct lookup *kinds* the registry can carry. Each
/// kind gets a compile-time slot index in `[0, MAX_TABLE_KINDS)`.
pub const MAX_TABLE_KINDS: u32 = 8;

// ── Per-slot table-info record ─────────────────────────────────────────
//
// Field order in `TableInfoSlot` (each field 4 bytes on wasm32):
//   0: occupancy        (i32; 0 = unclaimed, 1 = claimed)
//   1: table_idx        (i32; assigned id from `tables_len` at claim time)
//   2: mults_base       (ptr; forward only — AD slots leave it null)
//   3: inv_cnst_off     (i32; constraints-section start of this table)
//   4: inv_wit_off      (i32; witness-section start of this table)
//   5: num_indices      (i32; matches VM `TableInfo.num_indices`)
//   6: num_values       (i32; matches VM `TableInfo.num_values` — Phase 2
//                        dispatches on this: 0 = width-1, 1 = width-2)
//   7: length           (i32; matches VM `TableInfo.length`)

/// Offsets of fields *within a single `TableInfoSlot`*.
pub const TABLE_INFO_OCCUPANCY_OFFSET: u32 = 0;
pub const TABLE_INFO_TABLE_IDX_OFFSET: u32 = 4;
pub const TABLE_INFO_MULTS_BASE_PTR_OFFSET: u32 = 8;
pub const TABLE_INFO_INV_CNST_OFF_OFFSET: u32 = 12;
pub const TABLE_INFO_INV_WIT_OFF_OFFSET: u32 = 16;
pub const TABLE_INFO_NUM_INDICES_OFFSET: u32 = 20;
pub const TABLE_INFO_NUM_VALUES_OFFSET: u32 = 24;
pub const TABLE_INFO_LENGTH_OFFSET: u32 = 28;
/// Total size of one `TableInfoSlot` (8 × 4 = 32 bytes on wasm32).
pub const TABLE_INFO_SLOT_SIZE: u32 = 32;

const _: () = assert!(TABLE_INFO_TABLE_IDX_OFFSET == TABLE_INFO_OCCUPANCY_OFFSET + 4);
const _: () = assert!(TABLE_INFO_MULTS_BASE_PTR_OFFSET == TABLE_INFO_TABLE_IDX_OFFSET + 4);
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
//   - mults_cursor (cursor into the witness multiplicities section; first-
//     use lookups snapshot it, then bump by their table's length)
//   - lookups_a, lookups_b, lookups_c (lookup-tape write cursors)
//   - inputs (immutable input base pointer)
//   - tables_len (count of claimed tables; first-use claim assigns it as
//     the new table id and then bumps it)
//   - current_cnst_tables_off (cursor into the tables region of the
//     constraints section)
//   - current_wit_tables_off (cursor into the post-commitment witness
//     tables section, relative to the witness base — not to challenges)
//
// Then the registry: `[TableInfoSlot; MAX_TABLE_KINDS]`.

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
pub const WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET: u32 = 40;
pub const WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET: u32 = 44;
/// Base of `[TableInfoSlot; MAX_TABLE_KINDS]`. Slot at offset
/// `WITGEN_TABLES_REGISTRY_OFFSET + slot * TABLE_INFO_SLOT_SIZE` from the
/// VM-struct base. Within the slot, individual fields use the
/// `TABLE_INFO_*_OFFSET` constants above.
pub const WITGEN_TABLES_REGISTRY_OFFSET: u32 = 48;
pub const WITGEN_VM_STRUCT_SIZE: u32 =
    WITGEN_TABLES_REGISTRY_OFFSET + TABLE_INFO_SLOT_SIZE * MAX_TABLE_KINDS;

// ── AD VM struct ────────────────────────────────────────────────────────
//
// Cursors (kind-agnostic):
//   - out_da, out_db, out_dc (immutable output bases)
//   - ad_coeffs (cursor — advances on NextDCoeff)
//   - current_wit_off (next fresh witness index)
//   - ad_coeffs_base (immutable base for random-access reads)
//   - current_lookup_wit_off (cursor for lookup-section witness writes)
//   - current_cnst_tables_off, current_wit_tables_off,
//     current_wit_multiplicities_off (table-region cursors; first-use
//     lookups snapshot and bump)
//
// Then the registry: `[TableInfoSlot; MAX_TABLE_KINDS]` — same shape as the
// witgen one, but only `occupancy` and `inv_cnst_off` are read by the AD
// per-call body (all others are written on first-use claim and otherwise
// unused on this side).

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
pub const AD_TABLES_REGISTRY_OFFSET: u32 = 40;
pub const AD_VM_STRUCT_SIZE: u32 =
    AD_TABLES_REGISTRY_OFFSET + TABLE_INFO_SLOT_SIZE * MAX_TABLE_KINDS;

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
const _: () = assert!(WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET == WITGEN_TABLES_LEN_OFFSET + 4);
const _: () =
    assert!(WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET == WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET + 4);
const _: () = assert!(WITGEN_TABLES_REGISTRY_OFFSET == WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET + 4);
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
const _: () = assert!(AD_TABLES_REGISTRY_OFFSET == AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET + 4);
