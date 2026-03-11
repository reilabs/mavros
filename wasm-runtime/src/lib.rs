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

// ═══════════════════════════════════════════════════════════════════════════════
// Heap allocation (delegates to Rust's global allocator, dlmalloc on wasm32)
//
// Each allocation prepends an 8-byte header storing the requested size so that
// free() can reconstruct the Layout needed by dealloc().
// ═══════════════════════════════════════════════════════════════════════════════

const HEADER: usize = 8;
const ALIGN: usize = 8;

#[no_mangle]
pub unsafe extern "C" fn malloc(size: u32) -> *mut u8 {
    let total = HEADER + size as usize;
    let layout = std::alloc::Layout::from_size_align_unchecked(total, ALIGN);
    let base = std::alloc::alloc(layout);
    if base.is_null() {
        return base;
    }
    *(base as *mut u32) = size;
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
    *witness_ptr_ptr = witness_ptr.add(4);
}

#[no_mangle]
pub unsafe extern "C" fn __write_a(vm_ptr: *mut u8, v0: i64, v1: i64, v2: i64, v3: i64) {
    // a_ptr is at offset 1 (8 bytes on 64-bit, 4 bytes on 32-bit wasm)
    let a_ptr_ptr = (vm_ptr as *mut *mut u64).add(1);
    let a_ptr = *a_ptr_ptr;
    write_field(a_ptr, limbs_to_fr(v0, v1, v2, v3));
    *a_ptr_ptr = a_ptr.add(4);
}

#[no_mangle]
pub unsafe extern "C" fn __write_b(vm_ptr: *mut u8, v0: i64, v1: i64, v2: i64, v3: i64) {
    // b_ptr is at offset 2
    let b_ptr_ptr = (vm_ptr as *mut *mut u64).add(2);
    let b_ptr = *b_ptr_ptr;
    write_field(b_ptr, limbs_to_fr(v0, v1, v2, v3));
    *b_ptr_ptr = b_ptr.add(4);
}

#[no_mangle]
pub unsafe extern "C" fn __write_c(vm_ptr: *mut u8, v0: i64, v1: i64, v2: i64, v3: i64) {
    // c_ptr is at offset 3
    let c_ptr_ptr = (vm_ptr as *mut *mut u64).add(3);
    let c_ptr = *c_ptr_ptr;
    write_field(c_ptr, limbs_to_fr(v0, v1, v2, v3));
    *c_ptr_ptr = c_ptr.add(4);
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
// AD (Automatic Differentiation) runtime functions
//
// AD VM struct layout (wasm32 offsets):
//   Offset 0:  out_da       (ptr to dA array, not advanced)
//   Offset 4:  out_db       (ptr to dB array)
//   Offset 8:  out_dc       (ptr to dC array)
//   Offset 12: ad_coeffs    (ptr, advanced on each read)
//   Offset 16: current_wit_off (i32, next witness index)
// ═══════════════════════════════════════════════════════════════════════════════

const AD_VM_OUT_DA: usize = 0;
const AD_VM_OUT_DB: usize = 1;
const AD_VM_OUT_DC: usize = 2;
const AD_VM_COEFFS: usize = 3;
const AD_VM_WIT_OFF: usize = 4;

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
    *coeffs_ptr_ptr = coeffs_ptr.add(4);
}

/// Allocate a fresh AD witness index, returns the index as i32.
#[no_mangle]
pub unsafe extern "C" fn __ad_fresh_witness_index(vm_ptr: *mut u8) -> i32 {
    let wit_off_ptr = (vm_ptr as *mut i32).add(AD_VM_WIT_OFF);
    let index = *wit_off_ptr;
    *wit_off_ptr = index + 1;
    index
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
    field_accum(da_ptr.add(index as usize * 4), s);
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
    field_accum(db_ptr.add(index as usize * 4), s);
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
    field_accum(dc_ptr.add(index as usize * 4), s);
}
