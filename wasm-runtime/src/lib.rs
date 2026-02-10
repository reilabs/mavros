//! Runtime Library for Mavros WASM
//!
//! Provides BN254 field arithmetic and VM write functions called by LLVM-generated WASM.
//! Field elements are 4 x i64 limbs in Montgomery form.
//!
//! ABI (matching LLVM's wasm32 lowering of [4 x i64]):
//!   __field_mul(result_ptr, a0, a1, a2, a3, b0, b1, b2, b3)
//!   __write_*(vm_ptr, v0, v1, v2, v3)

use ark_bn254::Fr;
use ark_ff::BigInt;

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
    a0: i64, a1: i64, a2: i64, a3: i64,
    b0: i64, b1: i64, b2: i64, b3: i64,
) {
    let a = limbs_to_fr(a0, a1, a2, a3);
    let b = limbs_to_fr(b0, b1, b2, b3);
    write_field(result_ptr, a * b);
}

#[no_mangle]
pub unsafe extern "C" fn __write_witness(
    vm_ptr: *mut u8,
    v0: i64, v1: i64, v2: i64, v3: i64,
) {
    // VM struct: [witness_ptr, a_ptr, b_ptr, c_ptr] - 4 pointers
    // witness_ptr is at offset 0
    let witness_ptr_ptr = vm_ptr as *mut *mut u64;
    let witness_ptr = *witness_ptr_ptr;
    write_field(witness_ptr, limbs_to_fr(v0, v1, v2, v3));
    // Advance the pointer by 4 u64s (32 bytes)
    *witness_ptr_ptr = witness_ptr.add(4);
}

#[no_mangle]
pub unsafe extern "C" fn __write_a(
    vm_ptr: *mut u8,
    v0: i64, v1: i64, v2: i64, v3: i64,
) {
    // a_ptr is at offset 1 (8 bytes on 64-bit, 4 bytes on 32-bit wasm)
    let a_ptr_ptr = (vm_ptr as *mut *mut u64).add(1);
    let a_ptr = *a_ptr_ptr;
    write_field(a_ptr, limbs_to_fr(v0, v1, v2, v3));
    *a_ptr_ptr = a_ptr.add(4);
}

#[no_mangle]
pub unsafe extern "C" fn __write_b(
    vm_ptr: *mut u8,
    v0: i64, v1: i64, v2: i64, v3: i64,
) {
    // b_ptr is at offset 2
    let b_ptr_ptr = (vm_ptr as *mut *mut u64).add(2);
    let b_ptr = *b_ptr_ptr;
    write_field(b_ptr, limbs_to_fr(v0, v1, v2, v3));
    *b_ptr_ptr = b_ptr.add(4);
}

#[no_mangle]
pub unsafe extern "C" fn __write_c(
    vm_ptr: *mut u8,
    v0: i64, v1: i64, v2: i64, v3: i64,
) {
    // c_ptr is at offset 3
    let c_ptr_ptr = (vm_ptr as *mut *mut u64).add(3);
    let c_ptr = *c_ptr_ptr;
    write_field(c_ptr, limbs_to_fr(v0, v1, v2, v3));
    *c_ptr_ptr = c_ptr.add(4);
}
