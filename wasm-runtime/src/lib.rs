//! Runtime Library for Mavros WASM
//!
//! Provides BN254 field arithmetic and heap allocation called by
//! LLVM-generated WASM. Field elements are 4 x i64 limbs in Montgomery form.
//!
//! ABI (matching LLVM's wasm32 lowering of [4 x i64]):
//!   __field_mul(result_ptr, a0, a1, a2, a3, b0, b1, b2, b3)
//!
//! VM struct access (forward-pass writes, AD accumulators, AD witness/coeff
//! counters) is emitted inline in the generated LLVM module as GEP/load/store
//! against the vm_ptr — see `codegen/llssa_llvm_codegen.rs`.

// FIELD-ASSUMPTION: L1-direct-ref (1 sites)
use core::mem::MaybeUninit;

use ark_bn254::Fr;
use ark_ff::BigInt;

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

#[cfg(target_arch = "wasm32")]
const HEADER: usize = 8;
#[cfg(target_arch = "wasm32")]
const ALIGN: usize = 8;

#[cfg(target_arch = "wasm32")]
static mut LIVE_BYTES: usize = 0;

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __live_bytes() -> usize {
    unsafe { LIVE_BYTES }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: u32) -> *mut u8 {
    unsafe {
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
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(ptr: *mut u8) {
    #[allow(static_mut_refs)]
    unsafe {
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
    let limbs = fr.0.0;
    unsafe {
        *ptr = limbs[0];
        *ptr.add(1) = limbs[1];
        *ptr.add(2) = limbs[2];
        *ptr.add(3) = limbs[3];
    }
}

#[unsafe(no_mangle)]
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
    unsafe { write_field(result_ptr, a * b) };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field conversion
// ═══════════════════════════════════════════════════════════════════════════════

#[unsafe(no_mangle)]
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
    unsafe { write_field(result_ptr, fr) };
}

#[unsafe(no_mangle)]
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
    unsafe {
        *result_ptr = bigint.0[0];
        *result_ptr.add(1) = bigint.0[1];
        *result_ptr.add(2) = bigint.0[2];
        *result_ptr.add(3) = bigint.0[3];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field subtraction
// ═══════════════════════════════════════════════════════════════════════════════

#[unsafe(no_mangle)]
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
    unsafe { write_field(result_ptr, a - b) };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field addition
// ═══════════════════════════════════════════════════════════════════════════════

#[unsafe(no_mangle)]
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
    unsafe { write_field(result_ptr, a + b) };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field division
// ═══════════════════════════════════════════════════════════════════════════════

// FIELD-ASSUMPTION: L4-inverse
#[unsafe(no_mangle)]
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
    unsafe { write_field(result_ptr, result) };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Field less-than
// ═══════════════════════════════════════════════════════════════════════════════

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __field_lt(
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    b0: i64,
    b1: i64,
    b2: i64,
    b3: i64,
) -> bool {
    use ark_ff::PrimeField;
    let a = limbs_to_fr(a0, a1, a2, a3);
    let b = limbs_to_fr(b0, b1, b2, b3);
    a.into_bigint() < b.into_bigint()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wide integer helpers
//
// ABI, following the convention above: the operands are pointers, because one
// body serves every width and so cannot take an `iN` by value, and an integer
// result comes back through a pointer too. A comparison returns its answer as a
// `u32`, `1` or `0`.
//
//   __int_add(result_ptr, a_ptr, b_ptr, limbs)
//   __int_sub(result_ptr, a_ptr, b_ptr, limbs)
//   __int_mul(result_ptr, a_ptr, b_ptr, limbs)
//   __int_udivrem(quotient_ptr, remainder_ptr, a_ptr, b_ptr, limbs)
//   __int_sdivrem(quotient_ptr, remainder_ptr, a_ptr, b_ptr, limbs)
//   __int_shl(result_ptr, a_ptr, b_ptr, limbs, bits)
//   __int_lshr(result_ptr, a_ptr, b_ptr, limbs, bits)
//   __int_ashr(result_ptr, a_ptr, b_ptr, limbs, bits)
//   __int_spread(result_ptr, a_ptr, limbs, bits)
//   __int_unspread(odd_ptr, even_ptr, a_ptr, limbs, bits)
//   __int_eq(a_ptr, b_ptr, limbs) -> u32
//   __int_ult(a_ptr, b_ptr, limbs) -> u32
//   __int_slt(a_ptr, b_ptr, limbs, bits) -> u32
//
// Every buffer is `limbs` little-endian `u64`s, which is how LLVM lays out an `iN`
// on this target: the caller stores an `i64*limbs` value into the slot and loads
// the answer back. A helper that also takes `bits` reads its operand at that
// width: a shift wraps and reduces its amount there, `__int_slt` and `__int_ashr`
// take the sign from bit `bits - 1`, and a spread or unspread reads only the low
// `bits` bits. The signed division takes no `bits` and reads its operands at the
// whole `64 * limbs` bits, so the caller sign-extends them where the unsigned ones
// zero-extend.
//
// Why helpers: LLVM expands a wide `mul` into straight-line code quadratic in the
// width (7.2 MB at `i16384`, and past the wasm engine's function-size and
// local-count caps from about `i5700`); a loop over the limbs is one small
// function. Its wide `udiv` and `sdiv` are linear but large: unoptimized, about
// 20 000 wasm locals for a `udiv` at `i16384` and 28 000 to 36 000 for an
// `sdiv`, against the engine's cap of 50 000 per function. Even `add` and `sub`
// cost about 4 400 and 3 500 locals at `i16384`, where a helper call costs about
// 770 (the store and load of its buffers).
// ═══════════════════════════════════════════════════════════════════════════════

/// The low `limbs` limbs of `a + b`.
///
/// # Safety
///
/// `result` must be writable for `limbs` `u64`s, and `a` and `b` readable for `limbs` each.
/// `result` must not overlap either operand.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_add(result: *mut u64, a: *const u64, b: *const u64, limbs: u32) {
    let k = limbs as usize;
    unsafe {
        mavros_limb_arith::add(
            core::slice::from_raw_parts_mut(result, k),
            core::slice::from_raw_parts(a, k),
            core::slice::from_raw_parts(b, k),
        );
    }
}

/// The low `limbs` limbs of `a - b`.
///
/// # Safety
///
/// As [`__int_add`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_sub(result: *mut u64, a: *const u64, b: *const u64, limbs: u32) {
    let k = limbs as usize;
    unsafe {
        mavros_limb_arith::sub(
            core::slice::from_raw_parts_mut(result, k),
            core::slice::from_raw_parts(a, k),
            core::slice::from_raw_parts(b, k),
        );
    }
}

/// The low `limbs` limbs of `a * b`, by [`mavros_limb_arith::mul`].
///
/// # Safety
///
/// `result` must be writable for `limbs` `u64`s, and `a` and `b` readable for `limbs` each.
/// `result` must not overlap either operand; the backend gives each of the three a separate
/// scratch buffer, as `the_scratch_buffers_are_shared_and_sit_in_the_entry_block` checks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_mul(result: *mut u64, a: *const u64, b: *const u64, limbs: u32) {
    let k = limbs as usize;
    unsafe {
        mavros_limb_arith::mul(
            core::slice::from_raw_parts_mut(result, k),
            core::slice::from_raw_parts(a, k),
            core::slice::from_raw_parts(b, k),
        );
    }
}

/// The widest operand any helper here is handed, in limbs.
///
/// `MAX_BITS` in `mavros-int-semantics` caps every integer at 16384 bits, so the divisions keep
/// their scratch in stack arrays sized from this rather than allocating.
/// `the_limb_cap_is_the_type_cap` ties the two caps together.
const MAX_LIMBS: usize = 256;

/// The first `len` limbs of a stack buffer, zeroed, as a slice.
///
/// A buffer is sized for the widest operand, and a call initializes only the limbs it uses.
fn zeroed(store: &mut [MaybeUninit<u64>], len: usize) -> &mut [u64] {
    let prefix = &mut store[..len];
    for limb in prefix.iter_mut() {
        limb.write(0);
    }
    // SAFETY: every element of `prefix` was just initialized, and `MaybeUninit<u64>` has the
    // layout of `u64`.
    unsafe { &mut *(core::ptr::from_mut(prefix) as *mut [u64]) }
}

/// `a / b` and `a % b`, both read as unsigned, written into `quotient` and `remainder`.
///
/// Total: a zero divisor answers zero for both results, as the VM's `int_limbs::udivrem` does. A
/// division by a constant zero that codegen folds rather than calling this is LLVM poison instead.
/// The model leaves the case unspecified, because the guard IR never lets a division see a zero
/// divisor: a guarded one branches around it and an unguarded one asserts first.
///
/// # Safety
///
/// `quotient` and `remainder` must be writable for `limbs` `u64`s, and `a` and `b` readable for
/// `limbs` each. No buffer may overlap another, and `limbs` must be at most [`MAX_LIMBS`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_udivrem(
    quotient: *mut u64,
    remainder: *mut u64,
    a: *const u64,
    b: *const u64,
    limbs: u32,
) {
    let k = limbs as usize;
    unsafe {
        udivrem(
            core::slice::from_raw_parts_mut(quotient, k),
            core::slice::from_raw_parts_mut(remainder, k),
            core::slice::from_raw_parts(a, k),
            core::slice::from_raw_parts(b, k),
        );
    }
}

/// `a / b` and `a % b`, both read as two's complement at the whole `64 * limbs` bits.
///
/// Total, as [`__int_udivrem`] is: a zero divisor answers zero for both. The one other input the
/// model leaves unspecified, `INT_MIN / -1`, wraps to `INT_MIN` remainder zero.
///
/// # Safety
///
/// As [`__int_udivrem`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_sdivrem(
    quotient: *mut u64,
    remainder: *mut u64,
    a: *const u64,
    b: *const u64,
    limbs: u32,
) {
    let k = limbs as usize;
    unsafe {
        sdivrem(
            core::slice::from_raw_parts_mut(quotient, k),
            core::slice::from_raw_parts_mut(remainder, k),
            core::slice::from_raw_parts(a, k),
            core::slice::from_raw_parts(b, k),
        );
    }
}

/// The body of [`__int_udivrem`], over slices of one length, by [`mavros_limb_arith::udivrem`].
fn udivrem(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64]) {
    let mut store = [MaybeUninit::<u64>::uninit(); 2 * MAX_LIMBS + 1];
    let scratch = zeroed(&mut store, 2 * a.len() + 1);
    mavros_limb_arith::udivrem(quotient, remainder, a, b, scratch);
}

/// The body of [`__int_sdivrem`], by [`mavros_limb_arith::sdivrem`] at the buffer's whole width.
fn sdivrem(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64]) {
    let mut store = [MaybeUninit::<u64>::uninit(); 4 * MAX_LIMBS + 1];
    let scratch = zeroed(&mut store, 4 * a.len() + 1);
    mavros_limb_arith::sdivrem(quotient, remainder, a, b, 64 * a.len(), scratch);
}

/// The low `bits` bits of `a` spread into `result`, bit `i` to bit `2i`, by
/// [`mavros_limb_arith::spread`]. Every other bit of `result` is cleared and no bit of `a` above
/// `bits` is read.
///
/// # Safety
///
/// `result` must be writable and `a` readable for `limbs` `u64`s each, and `limbs` must hold
/// `2 * bits` bits. The two must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_spread(result: *mut u64, a: *const u64, limbs: u32, bits: u32) {
    let k = limbs as usize;
    unsafe {
        mavros_limb_arith::spread(
            core::slice::from_raw_parts_mut(result, k),
            core::slice::from_raw_parts(a, k),
            bits as usize,
        );
    }
}

/// The low `bits` bits of `a` separated into its odd- and even-indexed streams, by
/// [`mavros_limb_arith::unspread`]. Every other bit of both is cleared and no bit of `a` above
/// `bits` is read.
///
/// # Safety
///
/// `odd` and `even` must be writable and `a` readable for `limbs` `u64`s each, and `limbs` must hold
/// `bits` bits. No buffer may overlap another.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_unspread(
    odd: *mut u64,
    even: *mut u64,
    a: *const u64,
    limbs: u32,
    bits: u32,
) {
    let k = limbs as usize;
    unsafe {
        mavros_limb_arith::unspread(
            core::slice::from_raw_parts_mut(odd, k),
            core::slice::from_raw_parts_mut(even, k),
            core::slice::from_raw_parts(a, k),
            bits as usize,
        );
    }
}

/// `a` shifted by `b` at `bits`, by the [`mavros_limb_arith`] shift `shift` names, with `b` read as
/// unsigned and reduced modulo `bits` first.
///
/// # Safety
///
/// As [`__int_add`], and `bits` must be at least one and `limbs` must be `ceil(bits / 64)`. Every
/// bit of `b` above `bits` must be clear, as its whole buffer is the amount reduced; `a`'s are
/// masked off.
unsafe fn shift_at_width(
    shift: fn(&mut [u64], &[u64], usize, usize),
    result: *mut u64,
    a: *const u64,
    b: *const u64,
    limbs: u32,
    bits: u32,
) {
    let (k, bits) = (limbs as usize, bits as usize);
    unsafe {
        let b = core::slice::from_raw_parts(b, k);
        shift(
            core::slice::from_raw_parts_mut(result, k),
            core::slice::from_raw_parts(a, k),
            mavros_limb_arith::shift_amount(b, bits),
            bits,
        );
    }
}

/// `a << b` at `bits`, wrapping there, with `b` read as unsigned and reduced modulo `bits`.
///
/// # Safety
///
/// As [`shift_at_width`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_shl(
    result: *mut u64,
    a: *const u64,
    b: *const u64,
    limbs: u32,
    bits: u32,
) {
    unsafe { shift_at_width(mavros_limb_arith::shl, result, a, b, limbs, bits) }
}

/// `a >> b` at `bits`, zero-filling, with `b` read as unsigned and reduced modulo `bits`.
///
/// # Safety
///
/// As [`shift_at_width`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_lshr(
    result: *mut u64,
    a: *const u64,
    b: *const u64,
    limbs: u32,
    bits: u32,
) {
    unsafe { shift_at_width(mavros_limb_arith::ushr, result, a, b, limbs, bits) }
}

/// `a >> b` at `bits`, sign-filling under a two's complement reading there, with `b` read as
/// unsigned and reduced modulo `bits`.
///
/// # Safety
///
/// As [`shift_at_width`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_ashr(
    result: *mut u64,
    a: *const u64,
    b: *const u64,
    limbs: u32,
    bits: u32,
) {
    unsafe { shift_at_width(mavros_limb_arith::ashr, result, a, b, limbs, bits) }
}

/// Whether `a == b`, as `1` or `0`.
///
/// # Safety
///
/// `a` and `b` must be readable for `limbs` `u64`s each, with every bit above the width they are
/// compared at clear, as the whole of both buffers is compared.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_eq(a: *const u64, b: *const u64, limbs: u32) -> u32 {
    let k = limbs as usize;
    unsafe { u32::from(core::slice::from_raw_parts(a, k) == core::slice::from_raw_parts(b, k)) }
}

/// Whether `a < b` reading both as unsigned, as `1` or `0`.
///
/// # Safety
///
/// As [`__int_eq`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_ult(a: *const u64, b: *const u64, limbs: u32) -> u32 {
    let k = limbs as usize;
    unsafe {
        u32::from(mavros_limb_arith::ult(
            core::slice::from_raw_parts(a, k),
            core::slice::from_raw_parts(b, k),
        ))
    }
}

/// Whether `a < b` reading both as two's complement at `bits`, as `1` or `0`.
///
/// # Safety
///
/// As [`__int_eq`], and `bits` must be at least one and `limbs` must be `ceil(bits / 64)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __int_slt(a: *const u64, b: *const u64, limbs: u32, bits: u32) -> u32 {
    let k = limbs as usize;
    unsafe {
        u32::from(mavros_limb_arith::slt(
            core::slice::from_raw_parts(a, k),
            core::slice::from_raw_parts(b, k),
            bits as usize,
        ))
    }
}

/// The runtime helpers' conformance to the normative model in `mavros-int-semantics`.
///
/// Only the LLVM backend calls these helpers, so they conform under its `llvm` tag and its
/// relation: equal to [`residue`](mavros_int_semantics::residue) wherever the model has an answer,
/// which is everywhere except a division by zero and a signed `INT_MIN / -1`. A comparison is held
/// to [`IntBits::compare`](mavros_int_semantics::IntBits::compare).
///
/// The sweep also covers narrow widths, which the backend never routes here: one and two limbs are
/// where a limb-count error shows most plainly.
#[cfg(test)]
mod int_semantics_conformance {
    use mavros_int_semantics::{
        CmpOp, IntBits, IntOp, MAX_BITS, corners, int_bits::HOST_LIMB_BITS, residue,
    };

    /// [`super::__int_mul`] applied to two patterns of the same width.
    ///
    /// The buffers are the patterns' limbs, as in the backend, which stores an `iN` into a slot of
    /// `ceil(N / 64)` limbs and reads the answer back out of one.
    fn helper_mul(a: &IntBits, b: &IntBits) -> IntBits {
        assert_eq!(a.bits(), b.bits());
        let limbs = a.limb_count();
        let mut out = vec![0u64; limbs];
        unsafe {
            super::__int_mul(
                out.as_mut_ptr(),
                a.limbs().as_ptr(),
                b.limbs().as_ptr(),
                limbs as u32,
            );
        }
        IntBits::from_limbs(a.bits(), &out)
    }

    /// One of the four divisions through its helper, as the backend calls it.
    ///
    /// Each operand is widened to whole limbs, by sign extension for a signed division because the
    /// signed helper reads the sign at the top of the buffer, and the answer is truncated back to
    /// the operation's width.
    fn helper_divide(op: IntOp, a: &IntBits, b: &IntBits) -> IntBits {
        assert_eq!(a.bits(), b.bits());
        let bits = a.bits();
        let padded = a.limb_count() * HOST_LIMB_BITS;
        let signed = matches!(op, IntOp::SDiv | IntOp::SRem);
        let widen = |x: &IntBits| {
            if signed {
                x.sign_extend(padded)
            } else {
                x.cast(padded)
            }
        };
        let (a, b) = (widen(a), widen(b));

        let limbs = a.limb_count();
        let (mut quotient, mut remainder) = (vec![0u64; limbs], vec![0u64; limbs]);
        let helper = if signed {
            super::__int_sdivrem
        } else {
            super::__int_udivrem
        };
        unsafe {
            helper(
                quotient.as_mut_ptr(),
                remainder.as_mut_ptr(),
                a.limbs().as_ptr(),
                b.limbs().as_ptr(),
                limbs as u32,
            );
        }

        let answer = match op {
            IntOp::UDiv | IntOp::SDiv => quotient,
            IntOp::URem | IntOp::SRem => remainder,
            other => unreachable!("{other:?} is not a division"),
        };
        IntBits::from_limbs(bits, &answer)
    }

    const DIVISIONS: [IntOp; 4] = [IntOp::UDiv, IntOp::URem, IntOp::SDiv, IntOp::SRem];

    #[test]
    fn the_helper_divisions_agree_with_the_model() {
        let mut checked = 0usize;
        let mut at_the_widest = 0usize;

        let mut widths = vec![1usize, 2, 7, 8, 63, 64, 65, 96, 127, 128];
        widths.extend(corners::WIDE_WIDTHS);
        let widest = *corners::WIDE_WIDTHS
            .last()
            .expect("the wide set is not empty");

        for op in DIVISIONS {
            for &bits in &widths {
                let (lhs, rhs) = corners::wide_operands(op, bits);
                for a in &lhs {
                    for b in &rhs {
                        let Some(want) = residue(op, a, b) else {
                            continue;
                        };
                        let got = helper_divide(op, a, b);
                        assert_eq!(
                            got, want,
                            "{op:?} at {bits} bits: {a:?}, {b:?} gave {got:?}, model says {want:?}"
                        );
                        checked += 1;
                        if bits == widest {
                            at_the_widest += 1;
                        }
                    }
                }
            }
        }

        assert!(
            checked > 10_000,
            "the sweep only reached {checked} specified points"
        );
        assert!(
            at_the_widest > 400,
            "the widest width contributed only {at_the_widest} points"
        );
    }

    /// The inputs the model leaves unspecified get the VM's answer: a zero divisor gives zero
    /// for both, and a signed `INT_MIN / -1` wraps to `INT_MIN` with remainder zero. Guard IR keeps
    /// these from every division; the helper is total rather than trapping.
    #[test]
    fn the_helper_divisions_are_total() {
        for bits in [129, 1000, 16384] {
            for a in corners::wide_values(bits) {
                let zero = IntBits::zero(bits);
                for op in DIVISIONS {
                    assert!(
                        helper_divide(op, &a, &zero).is_zero(),
                        "{op:?} by zero at {bits} bits gave a non-zero answer"
                    );
                }
            }

            let min = IntBits::from_signed(bits, &IntBits::signed_min(bits));
            let minus_one = IntBits::all_ones(bits);
            assert_eq!(helper_divide(IntOp::SDiv, &min, &minus_one), min);
            assert!(helper_divide(IntOp::SRem, &min, &minus_one).is_zero());
        }
    }

    /// The add-back path of Knuth D, on Hacker's Delight's `divmnu` vector scaled to 64-bit limbs:
    /// `(2^191 + 3) / (2^189 + 1)`, where the estimate is 4 and the true quotient 3. The corner
    /// sweep reaches this path only by chance.
    #[test]
    fn a_quotient_limb_estimated_one_too_large_is_added_back() {
        let bits = 192;
        let a = IntBits::from_limbs(bits, &[3, 0, 1 << 63]);
        let b = IntBits::from_limbs(bits, &[1, 0, 1 << 61]);
        for op in [IntOp::UDiv, IntOp::URem] {
            assert_eq!(
                helper_divide(op, &a, &b),
                residue(op, &a, &b).expect("a non-zero divisor is specified"),
                "{op:?} through the add-back"
            );
        }
        assert_eq!(
            helper_divide(IntOp::UDiv, &a, &b),
            IntBits::from_limbs(bits, &[3])
        );
    }

    /// The divisions' stack buffers hold the widest operand the type system admits.
    #[test]
    fn the_limb_cap_is_the_type_cap() {
        assert_eq!(super::MAX_LIMBS, IntBits::limbs_for_bits(MAX_BITS));
    }

    /// [`super::__int_add`] or [`super::__int_sub`] applied to two patterns of the same width, the
    /// way [`helper_mul`] applies the multiply.
    fn helper_sum(op: IntOp, a: &IntBits, b: &IntBits) -> IntBits {
        assert_eq!(a.bits(), b.bits());
        let limbs = a.limb_count();
        let mut out = vec![0u64; limbs];
        let helper = match op {
            IntOp::UAdd | IntOp::SAdd => super::__int_add,
            IntOp::USub | IntOp::SSub => super::__int_sub,
            other => unreachable!("{other:?} is not a sum or difference"),
        };
        unsafe {
            helper(
                out.as_mut_ptr(),
                a.limbs().as_ptr(),
                b.limbs().as_ptr(),
                limbs as u32,
            );
        }
        IntBits::from_limbs(a.bits(), &out)
    }

    /// Both readings of both operations, because one helper serves each pair: a wrapping sum or
    /// difference has the same bits whichever way its operands are read, and the model's residue
    /// for an overflow says so.
    #[test]
    fn the_helper_sums_and_differences_agree_with_the_model() {
        let mut checked = 0usize;
        let mut at_the_widest = 0usize;

        let mut widths = vec![1usize, 2, 7, 8, 63, 64, 65, 96, 127, 128];
        widths.extend(corners::WIDE_WIDTHS);
        let widest = *corners::WIDE_WIDTHS
            .last()
            .expect("the wide set is not empty");

        for op in [IntOp::UAdd, IntOp::SAdd, IntOp::USub, IntOp::SSub] {
            for &bits in &widths {
                let (lhs, rhs) = corners::wide_operands(op, bits);
                for a in &lhs {
                    for b in &rhs {
                        let want = residue(op, a, b)
                            .expect("a sum or difference wraps, so the model always answers");
                        let got = helper_sum(op, a, b);
                        assert_eq!(
                            got, want,
                            "{op:?} at {bits} bits: {a:?}, {b:?} gave {got:?}, model says {want:?}"
                        );
                        checked += 1;
                        if bits == widest {
                            at_the_widest += 1;
                        }
                    }
                }
            }
        }

        assert!(
            checked > 10_000,
            "the sweep only reached {checked} specified points"
        );
        assert!(
            at_the_widest > 400,
            "the widest width contributed only {at_the_widest} points"
        );
    }

    /// [`super::__int_shl`], [`super::__int_lshr`] or [`super::__int_ashr`] applied to a value and
    /// an amount of the same width, the way the backend calls them: zero-extended to whole limbs,
    /// which keeps both the amount's unsigned value and the value's sign bit where the helper reads
    /// them.
    fn helper_shift(op: IntOp, a: &IntBits, b: &IntBits) -> IntBits {
        assert_eq!(a.bits(), b.bits());
        let limbs = a.limb_count();
        let mut out = vec![0u64; limbs];
        let helper = match op {
            IntOp::Shl => super::__int_shl,
            IntOp::UShr => super::__int_lshr,
            IntOp::SShr => super::__int_ashr,
            other => unreachable!("{other:?} is not a shift"),
        };
        unsafe {
            helper(
                out.as_mut_ptr(),
                a.limbs().as_ptr(),
                b.limbs().as_ptr(),
                limbs as u32,
                a.bits() as u32,
            );
        }
        IntBits::from_limbs(a.bits(), &out)
    }

    /// [`super::__int_eq`], [`super::__int_ult`] or [`super::__int_slt`] applied to two patterns of
    /// the same width.
    fn helper_compare(op: CmpOp, a: &IntBits, b: &IntBits) -> bool {
        assert_eq!(a.bits(), b.bits());
        let limbs = a.limb_count() as u32;
        let (a_limbs, b_limbs) = (a.limbs().as_ptr(), b.limbs().as_ptr());
        let answer = unsafe {
            match op {
                CmpOp::Eq => super::__int_eq(a_limbs, b_limbs, limbs),
                CmpOp::ULt => super::__int_ult(a_limbs, b_limbs, limbs),
                CmpOp::SLt => super::__int_slt(a_limbs, b_limbs, limbs, a.bits() as u32),
            }
        };
        assert!(answer <= 1, "{op:?} answered {answer}, which is not a bit");
        answer == 1
    }

    /// The three shifts over every amount the model sweeps, an over-wide one included since the
    /// helper reduces it, and the three orderings over every corner pair.
    #[test]
    fn the_helper_shifts_and_orderings_agree_with_the_model() {
        let (mut shifted, mut ordered) = (0usize, 0usize);
        let mut widths = vec![1usize, 2, 7, 8, 63, 64, 65, 96, 127, 128];
        widths.extend(corners::WIDE_WIDTHS);

        for &bits in &widths {
            for op in [IntOp::Shl, IntOp::UShr, IntOp::SShr] {
                let values = corners::wide_values(bits);
                for a in &values {
                    for b in &corners::wide_shift_amounts(bits) {
                        let want = residue(op, a, b).expect("a shift always has a residue");
                        let got = helper_shift(op, a, b);
                        assert_eq!(
                            got, want,
                            "{op:?} at {bits} bits: {a:?} by {b:?} gave {got:?}, model says {want:?}"
                        );
                        shifted += 1;
                    }
                }
            }
            let values = corners::wide_values(bits);
            for a in &values {
                for b in &values {
                    for op in CmpOp::ALL {
                        assert_eq!(
                            helper_compare(op, a, b),
                            a.compare(op, b),
                            "{op:?} at {bits} bits: {a:?}, {b:?}"
                        );
                        ordered += 1;
                    }
                }
            }
        }
        assert!(shifted > 10_000, "the shifts only reached {shifted} points");
        assert!(
            ordered > 10_000,
            "the orderings only reached {ordered} points"
        );
    }

    #[test]
    fn the_helper_multiply_agrees_with_the_model() {
        let mut checked = 0usize;
        let mut at_the_widest = 0usize;

        // Every wide width the backend can route here, plus the narrow ones it never will.
        let mut widths = vec![1usize, 2, 7, 8, 63, 64, 65, 96, 127, 128];
        widths.extend(corners::WIDE_WIDTHS);

        let widest = *corners::WIDE_WIDTHS
            .last()
            .expect("the wide set is not empty");

        for bits in widths {
            // A multiply's two operand lists are the same; `wide_operands` builds each 256-limb
            // pattern once.
            let (lhs, rhs) = corners::wide_operands(IntOp::UMul, bits);
            for a in &lhs {
                for b in &rhs {
                    let want = residue(IntOp::UMul, a, b)
                        .expect("a multiply is total, so the model always answers");
                    let got = helper_mul(a, b);
                    assert_eq!(
                        got, want,
                        "__int_mul at {bits} bits: {a:?} * {b:?} gave {got:?}, model says {want:?}"
                    );
                    checked += 1;
                    if bits == widest {
                        at_the_widest += 1;
                    }
                }
            }
        }

        // Without this a helper that was never called would pass every assertion above.
        assert!(
            checked > 3_500,
            "the sweep only reached {checked} specified points"
        );

        // Without this, the narrow half alone could meet the count above.
        assert!(
            at_the_widest > 100,
            "the widest width contributed only {at_the_widest} points"
        );
    }

    /// `__int_mul` itself truncates past the width, not [`IntBits`]'s normalization: a product
    /// that overflows a width half-filling its top limb.
    #[test]
    fn a_product_that_overflows_the_width_is_truncated_to_it() {
        let bits = 129;
        let a = IntBits::from_biguint(bits, &(num_bigint::BigUint::from(1u8) << 128));
        let want = residue(IntOp::UMul, &a, &a).expect("a multiply is total");

        assert!(
            want.is_zero(),
            "2^128 squared is 2^256, which is 0 mod 2^129"
        );
        assert_eq!(helper_mul(&a, &a), want);
    }

    /// [`super::__int_spread`] and [`super::__int_unspread`] against the model, through buffers
    /// laid out as the backend lays them out: a spread's operand zero-extended into the result's
    /// whole limbs, an unspread's streams in the operand's.
    ///
    /// Every output buffer starts as garbage and every answer is compared limb for limb, so a limb
    /// the helper failed to clear shows. The widest widths each op takes are asserted present, as a
    /// filter that dropped them would leave the sweep green.
    #[test]
    fn the_helper_spreads_and_unspreads_agree_with_the_model() {
        const GARBAGE: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        let mut widths = vec![1usize, 2, 7, 8, 63, 64, 65, 96, 127, 128];
        widths.extend(corners::WIDE_WIDTHS);

        // A spread doubles its operand, so its widest is half the cap rather than the cap.
        let mut spread_widths: Vec<usize> = widths
            .iter()
            .copied()
            .filter(|&bits| bits <= MAX_BITS / 2)
            .collect();
        spread_widths.push(MAX_BITS / 2);
        for &bits in &spread_widths {
            for value in corners::wide_values(bits) {
                for read in corners::spread_reads(bits, 1) {
                    let padded = IntBits::limbs_for_bits(2 * bits) * HOST_LIMB_BITS;
                    let a = value.cast(padded);
                    let mut out = vec![GARBAGE; a.limb_count()];
                    unsafe {
                        super::__int_spread(
                            out.as_mut_ptr(),
                            a.limbs().as_ptr(),
                            a.limb_count() as u32,
                            read as u32,
                        );
                    }
                    assert_eq!(
                        out,
                        value.spread(read).cast(padded).limbs(),
                        "spread of {read} bits of {value:?}"
                    );
                }
            }
        }

        let unspread_widths: Vec<usize> = widths.into_iter().filter(|&bits| bits >= 2).collect();
        for &bits in &unspread_widths {
            for value in corners::wide_values(bits) {
                for read in corners::spread_reads(bits, 2) {
                    let limbs = value.limb_count();
                    let (mut odd, mut even) = (vec![GARBAGE; limbs], vec![GARBAGE; limbs]);
                    unsafe {
                        super::__int_unspread(
                            odd.as_mut_ptr(),
                            even.as_mut_ptr(),
                            value.limbs().as_ptr(),
                            limbs as u32,
                            read as u32,
                        );
                    }
                    let (want_odd, want_even) = value.unspread(read);
                    let padded = limbs * HOST_LIMB_BITS;
                    assert_eq!(
                        (odd.as_slice(), even.as_slice()),
                        (
                            want_odd.cast(padded).limbs(),
                            want_even.cast(padded).limbs()
                        ),
                        "unspread of {read} bits of {value:?}"
                    );
                }
            }
        }

        assert!(spread_widths.contains(&(MAX_BITS / 2)));
        assert!(unspread_widths.contains(&MAX_BITS) && unspread_widths.contains(&(MAX_BITS - 1)));
    }
}
