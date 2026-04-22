//! Runtime Library for Mavros WASM
//!
//! Provides BN254 field arithmetic, VM write functions, and heap allocation
//! called by LLVM-generated WASM.
//! Field elements are 4 x i64 limbs in Montgomery form.
//!
//! ABI (matching LLVM's wasm32 lowering of [4 x i64]):
//!   __field_mul(result_ptr, a0, a1, a2, a3, b0, b1, b2, b3)
//!   __write_*(vm_ptr, v0, v1, v2, v3)

use ark_bn254::Fr;
use ark_ff::BigInt;

// Field-element sizes. Intentionally duplicated rather than imported from
// `mavros-artifacts` to keep the wasm-runtime dependency-light (this crate
// compiles to `wasm32-unknown-unknown` with `opt-level = "z"` + LTO; pulling
// in artifacts would drag serde, tracing, and arkworks metadata we don't need
// here). Keep in sync with `mavros_artifacts::{FIELD_BYTES, FIELD_LIMBS}`.
/// Size of one BN254 field element in bytes.
const FIELD_BYTES: usize = 32;
/// Number of u64 limbs in one BN254 field element.
const FIELD_LIMBS: usize = 4;

// ═══════════════════════════════════════════════════════════════════════════════
// Heap allocation (delegates to Rust's global allocator, dlmalloc on wasm32)
//
// Each allocation prepends an 8-byte header storing the requested size so that
// free() can reconstruct the Layout needed by dealloc().
//
// LIVE_BYTES tracks the bytes currently held by malloc.
// The host can read it via __live_bytes()
//
// ═══════════════════════════════════════════════════════════════════════════════

const HEADER: usize = 8;
const ALIGN: usize = 8;

static mut LIVE_BYTES: usize = 0;

#[no_mangle]
pub unsafe extern "C" fn __live_bytes() -> usize {
    LIVE_BYTES
}

#[no_mangle]
pub unsafe extern "C" fn malloc(size: u32) -> *mut u8 {
    let total = HEADER + size as usize;
    let layout = std::alloc::Layout::from_size_align_unchecked(total, ALIGN);
    let base = std::alloc::alloc(layout);
    if base.is_null() {
        return base;
    }
    *(base as *mut u32) = size;
    LIVE_BYTES += size as usize;
    base.add(HEADER)
}

#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let base = ptr.sub(HEADER);
    let size = *(base as *mut u32) as usize;
    let total = HEADER + size;
    let layout = std::alloc::Layout::from_size_align_unchecked(total, ALIGN);
    assert!(
        size <= LIVE_BYTES,
        "__live_bytes underflow: freeing {} bytes but only {} tracked",
        size,
        LIVE_BYTES
    );
    LIVE_BYTES -= size;
    std::alloc::dealloc(base, layout);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field arithmetic
// ═══════════════════════════════════════════════════════════════════════════════

#[inline]
fn limbs_to_fr(l0: i64, l1: i64, l2: i64, l3: i64) -> Fr {
    Fr::new_unchecked(BigInt::new([l0 as u64, l1 as u64, l2 as u64, l3 as u64]))
}

#[inline]
unsafe fn write_field(ptr: *mut u64, fr: Fr) {
    let limbs = fr.0 .0;
    *ptr = limbs[0];
    *ptr.add(1) = limbs[1];
    *ptr.add(2) = limbs[2];
    *ptr.add(3) = limbs[3];
}

#[no_mangle]
pub unsafe extern "C" fn __field_mul(
    result_ptr: *mut u64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    b0: i64,
    b1: i64,
    b2: i64,
    b3: i64,
) {
    let a = limbs_to_fr(a0, a1, a2, a3);
    let b = limbs_to_fr(b0, b1, b2, b3);
    write_field(result_ptr, a * b);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Witgen VM struct layout — keep in sync with `src/compiler/llssa_llvm_codegen.rs`.
// (Offsets are in u32 slots since each pointer is 4 bytes on wasm32.)
// ═══════════════════════════════════════════════════════════════════════════════

const WVM_WITNESS_CURSOR: usize = 0;
const WVM_A_CURSOR: usize = 1;
const WVM_B_CURSOR: usize = 2;
const WVM_C_CURSOR: usize = 3;
const WVM_LOOKUPS_A_CURSOR: usize = 4;
const WVM_LOOKUPS_B_CURSOR: usize = 5;
const WVM_LOOKUPS_C_CURSOR: usize = 6;
const WVM_WITNESS_PRE_BASE: usize = 7;
const WVM_WITNESS_POST_BASE: usize = 8;
const WVM_A_BASE: usize = 9;
const WVM_B_BASE: usize = 10;
const WVM_C_BASE: usize = 11;
const WVM_LOOKUPS_A_BASE: usize = 12;

// WitgenBuf tags — keep in sync with `witgen_buf_id` in the LLVM codegen.
const WBUF_WIT_PRE: i32 = 0;
const WBUF_WIT_POST: i32 = 1;
const WBUF_A: i32 = 2;
const WBUF_B: i32 = 3;
const WBUF_C: i32 = 4;

#[inline]
unsafe fn witgen_buf_base(vm_ptr: *mut u8, buf_id: i32) -> *mut u64 {
    let slot = match buf_id {
        WBUF_WIT_PRE => WVM_WITNESS_PRE_BASE,
        WBUF_WIT_POST => WVM_WITNESS_POST_BASE,
        WBUF_A => WVM_A_BASE,
        WBUF_B => WVM_B_BASE,
        WBUF_C => WVM_C_BASE,
        _ => core::hint::unreachable_unchecked(),
    };
    *(vm_ptr as *mut *mut u64).add(slot)
}

// ═══════════════════════════════════════════════════════════════════════════════
// VM write functions
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn __write_witness(vm_ptr: *mut u8, v0: i64, v1: i64, v2: i64, v3: i64) {
    // VM struct: [witness_ptr, a_ptr, b_ptr, c_ptr] - 4 pointers
    // witness_ptr is at offset 0
    let witness_ptr_ptr = vm_ptr as *mut *mut u64;
    let witness_ptr = *witness_ptr_ptr;
    write_field(witness_ptr, limbs_to_fr(v0, v1, v2, v3));
    // Advance the pointer by 4 u64s (32 bytes)
    *witness_ptr_ptr = witness_ptr.add(FIELD_LIMBS);
}

#[no_mangle]
pub unsafe extern "C" fn __write_a(vm_ptr: *mut u8, v0: i64, v1: i64, v2: i64, v3: i64) {
    // a_ptr is at offset 1 (8 bytes on 64-bit, 4 bytes on 32-bit wasm)
    let a_ptr_ptr = (vm_ptr as *mut *mut u64).add(1);
    let a_ptr = *a_ptr_ptr;
    write_field(a_ptr, limbs_to_fr(v0, v1, v2, v3));
    *a_ptr_ptr = a_ptr.add(FIELD_LIMBS);
}

#[no_mangle]
pub unsafe extern "C" fn __write_b(vm_ptr: *mut u8, v0: i64, v1: i64, v2: i64, v3: i64) {
    // b_ptr is at offset 2
    let b_ptr_ptr = (vm_ptr as *mut *mut u64).add(2);
    let b_ptr = *b_ptr_ptr;
    write_field(b_ptr, limbs_to_fr(v0, v1, v2, v3));
    *b_ptr_ptr = b_ptr.add(FIELD_LIMBS);
}

#[no_mangle]
pub unsafe extern "C" fn __write_c(vm_ptr: *mut u8, v0: i64, v1: i64, v2: i64, v3: i64) {
    // c_ptr is at offset 3
    let c_ptr_ptr = (vm_ptr as *mut *mut u64).add(3);
    let c_ptr = *c_ptr_ptr;
    write_field(c_ptr, limbs_to_fr(v0, v1, v2, v3));
    *c_ptr_ptr = c_ptr.add(FIELD_LIMBS);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 2 primitives — random-access on the witgen buffers, field inversion,
// and lookup-tape length.
// ═══════════════════════════════════════════════════════════════════════════════

/// Load `buf[idx]` as a Field (4 u64 limbs returned via `result_ptr` sret).
#[no_mangle]
pub unsafe extern "C" fn __witgen_load(
    result_ptr: *mut u64,
    vm_ptr: *mut u8,
    buf_id: i32,
    idx: i32,
) {
    let base = witgen_buf_base(vm_ptr, buf_id);
    let slot = base.add((idx as usize) * FIELD_LIMBS);
    let fr = limbs_to_fr(
        *slot as i64,
        *slot.add(1) as i64,
        *slot.add(2) as i64,
        *slot.add(3) as i64,
    );
    write_field(result_ptr, fr);
}

/// Store `buf[idx] = value`.
#[no_mangle]
pub unsafe extern "C" fn __witgen_store(
    vm_ptr: *mut u8,
    buf_id: i32,
    idx: i32,
    v0: i64,
    v1: i64,
    v2: i64,
    v3: i64,
) {
    let base = witgen_buf_base(vm_ptr, buf_id);
    let slot = base.add((idx as usize) * FIELD_LIMBS);
    write_field(slot, limbs_to_fr(v0, v1, v2, v3));
}

/// In-place accumulate: `buf[idx] += value` in field arithmetic.
#[no_mangle]
pub unsafe extern "C" fn __witgen_add(
    vm_ptr: *mut u8,
    buf_id: i32,
    idx: i32,
    v0: i64,
    v1: i64,
    v2: i64,
    v3: i64,
) {
    let base = witgen_buf_base(vm_ptr, buf_id);
    let slot = base.add((idx as usize) * FIELD_LIMBS);
    let old = limbs_to_fr(
        *slot as i64,
        *slot.add(1) as i64,
        *slot.add(2) as i64,
        *slot.add(3) as i64,
    );
    write_field(slot, old + limbs_to_fr(v0, v1, v2, v3));
}

/// In-place low-u64 add: `buf[idx].low_u64 += value` as *integer* arithmetic.
///
/// NOT field arithmetic: treats the slot's low limb as a plain u64 counter.
/// Used by Phase 1 to accumulate lookup multiplicities as raw counts; Phase 2
/// later re-encodes each slot into Montgomery form. Upper limbs are left
/// untouched (callers guarantee they're zero on first use).
#[no_mangle]
pub unsafe extern "C" fn __witgen_add_low_u64(
    vm_ptr: *mut u8,
    buf_id: i32,
    idx: i32,
    value: i64,
) {
    let base = witgen_buf_base(vm_ptr, buf_id);
    let slot = base.add((idx as usize) * FIELD_LIMBS);
    *slot = (*slot).wrapping_add(value as u64);
}

/// Compute `1 / x` in the prime field.
///
/// Phase 2 calls this on the product of all `(α - i)` terms for active table
/// entries; that product is non-zero whenever the prover is well-formed (α is
/// a random challenge). A zero input means something upstream is already
/// broken, so we panic rather than silently returning 0 and smuggling invalid
/// witness data through.
#[no_mangle]
pub unsafe extern "C" fn __field_inverse(result_ptr: *mut u64, v0: i64, v1: i64, v2: i64, v3: i64) {
    let x = limbs_to_fr(v0, v1, v2, v3);
    let inv = ark_ff::Field::inverse(&x).expect("__field_inverse: input is zero");
    write_field(result_ptr, inv);
}

/// Number of field-element entries written to the A stream of the lookup tape
/// (one per lookup call). Computed as `(cursor - base) / sizeof(Field)`.
///
/// The cursor starts equal to the base and can only advance during Phase 1,
/// so `cursor >= base` is an invariant. A violation means the VM struct got
/// corrupted — assert rather than wrap so the failure is obvious.
#[no_mangle]
pub unsafe extern "C" fn __lookup_tape_len(vm_ptr: *mut u8) -> i32 {
    let base_ptr = *(vm_ptr as *const *const u64).add(WVM_LOOKUPS_A_BASE);
    let cursor_ptr = *(vm_ptr as *const *const u64).add(WVM_LOOKUPS_A_CURSOR);
    let cursor = cursor_ptr as usize;
    let base = base_ptr as usize;
    assert!(
        cursor >= base,
        "__lookup_tape_len: cursor (0x{:x}) moved below base (0x{:x})",
        cursor,
        base
    );
    let bytes = cursor - base;
    (bytes / FIELD_BYTES) as i32
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field conversion
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn __field_from_limbs(
    result_ptr: *mut u64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
) {
    use ark_ff::PrimeField;
    let bigint = BigInt::new([a0 as u64, a1 as u64, a2 as u64, a3 as u64]);
    let fr = Fr::from_bigint(bigint).expect("Constructing a field from limbs failed");
    write_field(result_ptr, fr);
}

#[no_mangle]
pub unsafe extern "C" fn __field_to_limbs(
    result_ptr: *mut u64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
) {
    use ark_ff::PrimeField;
    let fr = limbs_to_fr(a0, a1, a2, a3);
    let bigint = fr.into_bigint();
    *result_ptr = bigint.0[0];
    *result_ptr.add(1) = bigint.0[1];
    *result_ptr.add(2) = bigint.0[2];
    *result_ptr.add(3) = bigint.0[3];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field subtraction
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn __field_sub(
    result_ptr: *mut u64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    b0: i64,
    b1: i64,
    b2: i64,
    b3: i64,
) {
    let a = limbs_to_fr(a0, a1, a2, a3);
    let b = limbs_to_fr(b0, b1, b2, b3);
    write_field(result_ptr, a - b);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field addition
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn __field_add(
    result_ptr: *mut u64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    b0: i64,
    b1: i64,
    b2: i64,
    b3: i64,
) {
    let a = limbs_to_fr(a0, a1, a2, a3);
    let b = limbs_to_fr(b0, b1, b2, b3);
    write_field(result_ptr, a + b);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field division
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn __field_div(
    result_ptr: *mut u64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    b0: i64,
    b1: i64,
    b2: i64,
    b3: i64,
) {
    use ark_ff::AdditiveGroup;
    let a = limbs_to_fr(a0, a1, a2, a3);
    let b = limbs_to_fr(b0, b1, b2, b3);
    let result = if b == Fr::ZERO { Fr::ZERO } else { a / b };
    write_field(result_ptr, result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// AD (Automatic Differentiation) runtime functions
//
// AD VM struct layout (wasm32 offsets):
//   Offset 0:  out_da       (ptr to dA array, not advanced)
//   Offset 4:  out_db       (ptr to dB array)
//   Offset 8:  out_dc       (ptr to dC array)
//   Offset 12: ad_coeffs    (cursor ptr, advanced by __ad_next_d_coeff)
//   Offset 16: current_wit_off (i32, next witness index)
//   Offset 20: ad_coeffs_base   (ptr, immutable; base for random-access reads)
//   Offset 24: ad_coeffs_tables (cursor ptr, advanced by __ad_next_d_coeff_tables)
//   Offset 28: ad_coeffs_lookups (cursor ptr, advanced by __ad_next_d_coeff_lookups)
// ═══════════════════════════════════════════════════════════════════════════════

const AD_VM_OUT_DA: usize = 0;
const AD_VM_OUT_DB: usize = 1;
const AD_VM_OUT_DC: usize = 2;
const AD_VM_COEFFS: usize = 3;
const AD_VM_WIT_OFF: usize = 4;
const AD_VM_COEFFS_BASE: usize = 5;
const AD_VM_COEFFS_TABLES: usize = 6;
const AD_VM_COEFFS_LOOKUPS: usize = 7;
const AD_VM_LOOKUP_WIT_OFF: usize = 8;

/// Read the next sensitivity coefficient from the AD input tape.
/// Returns 4 limbs via result pointer (sret convention on wasm32).
#[no_mangle]
pub unsafe extern "C" fn __ad_next_d_coeff(result_ptr: *mut u64, vm_ptr: *mut u8) {
    let coeffs_ptr_ptr = (vm_ptr as *mut *const u64).add(AD_VM_COEFFS);
    let coeffs_ptr = *coeffs_ptr_ptr;
    // Read 4 limbs
    let fr = limbs_to_fr(
        *coeffs_ptr as i64,
        *coeffs_ptr.add(1) as i64,
        *coeffs_ptr.add(2) as i64,
        *coeffs_ptr.add(3) as i64,
    );
    write_field(result_ptr, fr);
    // Advance by 4 u64s (one field element)
    *coeffs_ptr_ptr = coeffs_ptr.add(FIELD_LIMBS);
}

/// Read the next sensitivity coefficient from the AD tables-section cursor.
#[no_mangle]
pub unsafe extern "C" fn __ad_next_d_coeff_tables(result_ptr: *mut u64, vm_ptr: *mut u8) {
    let coeffs_ptr_ptr = (vm_ptr as *mut *const u64).add(AD_VM_COEFFS_TABLES);
    let coeffs_ptr = *coeffs_ptr_ptr;
    let fr = limbs_to_fr(
        *coeffs_ptr as i64,
        *coeffs_ptr.add(1) as i64,
        *coeffs_ptr.add(2) as i64,
        *coeffs_ptr.add(3) as i64,
    );
    write_field(result_ptr, fr);
    *coeffs_ptr_ptr = coeffs_ptr.add(FIELD_LIMBS);
}

/// Read the next sensitivity coefficient from the AD lookups-section cursor.
#[no_mangle]
pub unsafe extern "C" fn __ad_next_d_coeff_lookups(result_ptr: *mut u64, vm_ptr: *mut u8) {
    let coeffs_ptr_ptr = (vm_ptr as *mut *const u64).add(AD_VM_COEFFS_LOOKUPS);
    let coeffs_ptr = *coeffs_ptr_ptr;
    let fr = limbs_to_fr(
        *coeffs_ptr as i64,
        *coeffs_ptr.add(1) as i64,
        *coeffs_ptr.add(2) as i64,
        *coeffs_ptr.add(3) as i64,
    );
    write_field(result_ptr, fr);
    *coeffs_ptr_ptr = coeffs_ptr.add(FIELD_LIMBS);
}

/// Read an AD coefficient at a fixed absolute offset (in field elements).
/// Does not advance any cursor.
#[no_mangle]
pub unsafe extern "C" fn __ad_read_coeff_at(result_ptr: *mut u64, vm_ptr: *mut u8, offset: i32) {
    let base_ptr_ptr = (vm_ptr as *const *const u64).add(AD_VM_COEFFS_BASE);
    let base_ptr = *base_ptr_ptr;
    let slot_ptr = base_ptr.add((offset as usize) * FIELD_LIMBS);
    let fr = limbs_to_fr(
        *slot_ptr as i64,
        *slot_ptr.add(1) as i64,
        *slot_ptr.add(2) as i64,
        *slot_ptr.add(3) as i64,
    );
    write_field(result_ptr, fr);
}

/// Allocate a fresh AD witness index, returns the index as i32.
#[no_mangle]
pub unsafe extern "C" fn __ad_fresh_witness_index(vm_ptr: *mut u8) -> i32 {
    let wit_off_ptr = (vm_ptr as *mut i32).add(AD_VM_WIT_OFF);
    let index = *wit_off_ptr;
    *wit_off_ptr = index + 1;
    index
}

/// Return the next absolute witness offset for the LogUp lookup section and
/// advance the cursor. Used by AD lookup helpers.
#[no_mangle]
pub unsafe extern "C" fn __ad_next_lookup_wit_off(vm_ptr: *mut u8) -> i32 {
    let off_ptr = (vm_ptr as *mut i32).add(AD_VM_LOOKUP_WIT_OFF);
    let idx = *off_ptr;
    *off_ptr = idx + 1;
    idx
}

/// Helper: read field from memory, add value, write back.
#[inline]
unsafe fn field_accum(dst: *mut u64, v: Fr) {
    let old = limbs_to_fr(
        *dst as i64,
        *dst.add(1) as i64,
        *dst.add(2) as i64,
        *dst.add(3) as i64,
    );
    write_field(dst, old + v);
}

/// Accumulate sensitivity * const_value into out_da at position 0.
#[no_mangle]
pub unsafe extern "C" fn __ad_accum_da(
    vm_ptr: *mut u8,
    cv0: i64,
    cv1: i64,
    cv2: i64,
    cv3: i64,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) {
    let cv = limbs_to_fr(cv0, cv1, cv2, cv3);
    let s = limbs_to_fr(s0, s1, s2, s3);
    let da_ptr = *(vm_ptr as *mut *mut u64).add(AD_VM_OUT_DA);
    field_accum(da_ptr, s * cv);
}

/// Accumulate sensitivity * const_value into out_db at position 0.
#[no_mangle]
pub unsafe extern "C" fn __ad_accum_db(
    vm_ptr: *mut u8,
    cv0: i64,
    cv1: i64,
    cv2: i64,
    cv3: i64,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) {
    let cv = limbs_to_fr(cv0, cv1, cv2, cv3);
    let s = limbs_to_fr(s0, s1, s2, s3);
    let db_ptr = *(vm_ptr as *mut *mut u64).add(AD_VM_OUT_DB);
    field_accum(db_ptr, s * cv);
}

/// Accumulate sensitivity * const_value into out_dc at position 0.
#[no_mangle]
pub unsafe extern "C" fn __ad_accum_dc(
    vm_ptr: *mut u8,
    cv0: i64,
    cv1: i64,
    cv2: i64,
    cv3: i64,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) {
    let cv = limbs_to_fr(cv0, cv1, cv2, cv3);
    let s = limbs_to_fr(s0, s1, s2, s3);
    let dc_ptr = *(vm_ptr as *mut *mut u64).add(AD_VM_OUT_DC);
    field_accum(dc_ptr, s * cv);
}

/// Accumulate sensitivity into out_da at witness position index.
#[no_mangle]
pub unsafe extern "C" fn __ad_accum_at_da(
    vm_ptr: *mut u8,
    index: i32,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) {
    let s = limbs_to_fr(s0, s1, s2, s3);
    let da_ptr = *(vm_ptr as *mut *mut u64).add(AD_VM_OUT_DA);
    // Each field element is 4 u64s = 32 bytes
    field_accum(da_ptr.add(index as usize * FIELD_LIMBS), s);
}

/// Accumulate sensitivity into out_db at witness position index.
#[no_mangle]
pub unsafe extern "C" fn __ad_accum_at_db(
    vm_ptr: *mut u8,
    index: i32,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) {
    let s = limbs_to_fr(s0, s1, s2, s3);
    let db_ptr = *(vm_ptr as *mut *mut u64).add(AD_VM_OUT_DB);
    field_accum(db_ptr.add(index as usize * FIELD_LIMBS), s);
}

/// Accumulate sensitivity into out_dc at witness position index.
#[no_mangle]
pub unsafe extern "C" fn __ad_accum_at_dc(
    vm_ptr: *mut u8,
    index: i32,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) {
    let s = limbs_to_fr(s0, s1, s2, s3);
    let dc_ptr = *(vm_ptr as *mut *mut u64).add(AD_VM_OUT_DC);
    field_accum(dc_ptr.add(index as usize * FIELD_LIMBS), s);
}
