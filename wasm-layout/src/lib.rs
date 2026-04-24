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
// Layout (all pointers, in order):
//   0: witness cursor          (advances on each __write_witness)
//   1: a cursor                (advances on each __write_a)
//   2: b cursor                (advances on each __write_b)
//   3: c cursor                (advances on each __write_c)
//   4: multiplicities base     (immutable pointer to first slot of the first
//                               table's multiplicities — used by rangecheck
//                               helpers for random-access bumps)
//   5: lookup-tape a cursor    (one Field per Lookup call)
//   6: lookup-tape b cursor    (one Field per Lookup call)
//   7: lookup-tape c cursor    (one Field per Lookup call)
//   8: input base pointer      (immutable pointer to flattened ABI inputs)

pub const WITGEN_WITNESS_PTR_OFFSET: u32 = 0;
pub const WITGEN_A_PTR_OFFSET: u32 = 4;
pub const WITGEN_B_PTR_OFFSET: u32 = 8;
pub const WITGEN_C_PTR_OFFSET: u32 = 12;
pub const WITGEN_MULTS_BASE_PTR_OFFSET: u32 = 16;
pub const WITGEN_LOOKUPS_A_PTR_OFFSET: u32 = 20;
pub const WITGEN_LOOKUPS_B_PTR_OFFSET: u32 = 24;
pub const WITGEN_LOOKUPS_C_PTR_OFFSET: u32 = 28;
pub const WITGEN_INPUTS_PTR_OFFSET: u32 = 32;
pub const WITGEN_VM_STRUCT_SIZE: u32 = 36;

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

pub const AD_OUT_DA_PTR_OFFSET: u32 = 0;
pub const AD_OUT_DB_PTR_OFFSET: u32 = 4;
pub const AD_OUT_DC_PTR_OFFSET: u32 = 8;
pub const AD_COEFFS_PTR_OFFSET: u32 = 12;
pub const AD_CURRENT_WIT_OFF_OFFSET: u32 = 16;
pub const AD_COEFFS_BASE_PTR_OFFSET: u32 = 20;
pub const AD_CURRENT_LOOKUP_WIT_OFF_OFFSET: u32 = 24;
pub const AD_VM_STRUCT_SIZE: u32 = 28;

// Static shape checks — if someone renumbers a field, the build breaks here
// rather than at runtime in WASM.
const _: () = assert!(WITGEN_VM_STRUCT_SIZE == 9 * WASM_PTR_SIZE);
const _: () = assert!(AD_VM_STRUCT_SIZE == 6 * WASM_PTR_SIZE + 4);
const _: () = assert!(WITGEN_A_PTR_OFFSET == WITGEN_WITNESS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_B_PTR_OFFSET == WITGEN_A_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_C_PTR_OFFSET == WITGEN_B_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_MULTS_BASE_PTR_OFFSET == WITGEN_C_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_LOOKUPS_A_PTR_OFFSET == WITGEN_MULTS_BASE_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_LOOKUPS_B_PTR_OFFSET == WITGEN_LOOKUPS_A_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_LOOKUPS_C_PTR_OFFSET == WITGEN_LOOKUPS_B_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_INPUTS_PTR_OFFSET == WITGEN_LOOKUPS_C_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_OUT_DB_PTR_OFFSET == AD_OUT_DA_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_OUT_DC_PTR_OFFSET == AD_OUT_DB_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_COEFFS_PTR_OFFSET == AD_OUT_DC_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_CURRENT_WIT_OFF_OFFSET == AD_COEFFS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_COEFFS_BASE_PTR_OFFSET == AD_CURRENT_WIT_OFF_OFFSET + WASM_PTR_SIZE);
const _: () =
    assert!(AD_CURRENT_LOOKUP_WIT_OFF_OFFSET == AD_COEFFS_BASE_PTR_OFFSET + WASM_PTR_SIZE);
