//! Shared layout constants for the WASM VM structs.
//!
//! Three consumers must agree on these byte offsets:
//!   - the test harness in `src/bin/test_runner.rs` that populates the struct
//!     in WASM linear memory before invoking `mavros_main`;
//!   - `src/compiler/llssa_llvm_codegen.rs`, which emits calls that hand
//!     `vm_ptr` to the runtime helpers;
//!   - `wasm-runtime/src/lib.rs`, which dereferences `vm_ptr` inside those
//!     helpers.
//!
//! All offsets are in bytes. wasm32: pointers are 4 bytes, usize/i32 is 4 bytes.

#![no_std]

/// Size of a pointer in wasm32 linear memory (used to derive layout).
pub const WASM_PTR_SIZE: u32 = 4;

// ── Forward-pass (witgen) VM struct ─────────────────────────────────────
// 4 pointers, tightly packed.

pub const WITGEN_WITNESS_PTR_OFFSET: u32 = 0;
pub const WITGEN_A_PTR_OFFSET: u32 = 4;
pub const WITGEN_B_PTR_OFFSET: u32 = 8;
pub const WITGEN_C_PTR_OFFSET: u32 = 12;
pub const WITGEN_VM_STRUCT_SIZE: u32 = 16;

// ── AD VM struct ────────────────────────────────────────────────────────
// 4 pointers + one i32 witness counter.

pub const AD_OUT_DA_PTR_OFFSET: u32 = 0;
pub const AD_OUT_DB_PTR_OFFSET: u32 = 4;
pub const AD_OUT_DC_PTR_OFFSET: u32 = 8;
pub const AD_COEFFS_PTR_OFFSET: u32 = 12;
pub const AD_CURRENT_WIT_OFF_OFFSET: u32 = 16;
pub const AD_VM_STRUCT_SIZE: u32 = 20;

// Static shape checks — if someone renumbers a field, the build breaks here
// rather than at runtime in WASM.
const _: () = assert!(WITGEN_VM_STRUCT_SIZE == 4 * WASM_PTR_SIZE);
const _: () = assert!(AD_VM_STRUCT_SIZE == 4 * WASM_PTR_SIZE + 4);
const _: () = assert!(WITGEN_A_PTR_OFFSET == WITGEN_WITNESS_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_B_PTR_OFFSET == WITGEN_A_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(WITGEN_C_PTR_OFFSET == WITGEN_B_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_OUT_DB_PTR_OFFSET == AD_OUT_DA_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_OUT_DC_PTR_OFFSET == AD_OUT_DB_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_COEFFS_PTR_OFFSET == AD_OUT_DC_PTR_OFFSET + WASM_PTR_SIZE);
const _: () = assert!(AD_CURRENT_WIT_OFF_OFFSET == AD_COEFFS_PTR_OFFSET + WASM_PTR_SIZE);
