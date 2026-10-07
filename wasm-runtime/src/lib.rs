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
use mavros_limb_arith::{divide_by_limb, knuth_divide, negate, significant_limbs};

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
// ABI, following the convention above: the result comes back through a pointer,
// and the operands are pointers too because a width-generic helper cannot take
// an `iN` by value — one body serves every width, so the width arrives as the
// limb count `limbs` instead of in the type.
//
//   __int_add(result_ptr, a_ptr, b_ptr, limbs)
//   __int_sub(result_ptr, a_ptr, b_ptr, limbs)
//   __int_mul(result_ptr, a_ptr, b_ptr, limbs)
//   __int_udivrem(quotient_ptr, remainder_ptr, a_ptr, b_ptr, limbs)
//   __int_sdivrem(quotient_ptr, remainder_ptr, a_ptr, b_ptr, limbs)
//
// Every buffer is `limbs` little-endian `u64`s, which is exactly how LLVM lays
// an `iN` out in memory on this little-endian target: the caller stores an
// `i64*limbs` value into the slot and loads the answer back out of it. The
// signed division reads its operands at that whole width, so the caller
// sign-extends them into the slot where the unsigned ones zero-extend.
//
// LLVM's own expansion of a wide `mul` is straight-line code quadratic in the
// width -- 7.2 MB at `i16384`, and past the wasm engine's function-size and
// local-count caps from about `i5700` up -- where a loop over the limbs is one
// small function serving every width. Its wide `udiv` and `sdiv` are linear but
// not small: unoptimised codegen gives a bit-serial loop of about 20 000 wasm
// locals for a `udiv` at `i16384`, and 28 000 to 36 000 for an `sdiv`, against
// the engine's cap of 50 000 per function. Even its wide `add` and `sub`, linear
// and branch-free, cost about 4 400 and 3 500 locals at `i16384` unoptimised,
// where a call to a helper costs about 770: the store and load of its buffers.
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
/// `result` must not overlap either operand; the backend gives each of the three its own scratch
/// buffer, which is what `the_scratch_buffers_are_shared_and_sit_in_the_entry_block` counts.
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
/// `MAX_BITS` in `mavros-int-semantics` caps every integer type at 16384 bits, so no division is
/// wider than this, and the divisions keep their working copies in arrays of this size on their own
/// stack rather than allocating. `the_limb_cap_is_the_type_cap` ties the two together.
const MAX_LIMBS: usize = 256;

/// The first `len` limbs of a stack buffer, zeroed, as a slice.
///
/// A buffer is sized for the widest operand, and a call initializes only the limbs it uses.
fn zeroed(store: &mut [MaybeUninit<u64>], len: usize) -> &mut [u64] {
    let prefix = &mut store[..len];
    for limb in prefix.iter_mut() {
        limb.write(0);
    }
    // SAFETY: every element of `prefix` was just initialised, and `MaybeUninit<u64>` has the
    // layout of `u64`.
    unsafe { &mut *(core::ptr::from_mut(prefix) as *mut [u64]) }
}

/// `a / b` and `a % b`, both read as unsigned, written into `quotient` and `remainder`.
///
/// Total: a zero divisor answers zero for both results, as the VM's `int_limbs::udivrem` does. This
/// holds for a call; a division by a constant zero that codegen folds instead of calling this is
/// LLVM's poison. It is undefined behavior in LLVM's own `udiv`, and the model leaves it
/// unspecified because the guard IR keeps it from any division: a guarded one branches around it
/// and an unguarded one asserts first.
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

/// The body of [`__int_udivrem`], over slices of one length.
fn udivrem(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64]) {
    quotient.fill(0);
    remainder.fill(0);

    match significant_limbs(b) {
        0 => {}
        1 => remainder[0] = divide_by_limb(quotient, a, b[0]),
        n => {
            let mut divisor_store = [MaybeUninit::<u64>::uninit(); MAX_LIMBS];
            let mut dividend_store = [MaybeUninit::<u64>::uninit(); MAX_LIMBS + 1];
            let divisor = zeroed(&mut divisor_store, n);
            let dividend = zeroed(&mut dividend_store, a.len() + 1);
            knuth_divide(quotient, remainder, a, &b[..n], divisor, dividend);
        }
    }
}

/// The body of [`__int_sdivrem`]: sign-magnitude around [`udivrem`].
///
/// The quotient takes the operands' xor and the remainder takes the dividend's sign, which is
/// truncation toward zero.
fn sdivrem(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64]) {
    let k = a.len();
    let mut a_store = [MaybeUninit::<u64>::uninit(); MAX_LIMBS];
    let mut b_store = [MaybeUninit::<u64>::uninit(); MAX_LIMBS];
    let a_magnitude = zeroed(&mut a_store, k);
    let b_magnitude = zeroed(&mut b_store, k);
    a_magnitude.copy_from_slice(a);
    b_magnitude.copy_from_slice(b);

    let (a_negative, b_negative) = (is_negative(a), is_negative(b));
    if a_negative {
        negate(a_magnitude);
    }
    if b_negative {
        negate(b_magnitude);
    }

    udivrem(quotient, remainder, a_magnitude, b_magnitude);

    if a_negative != b_negative {
        negate(quotient);
    }
    if a_negative {
        negate(remainder);
    }
}

/// The top bit of the top limb, which is the sign of a value read at the buffer's whole width.
fn is_negative(value: &[u64]) -> bool {
    value.last().is_some_and(|&top| top >> 63 == 1)
}

/// The runtime helper's conformance relation to the normative model in `mavros-int-semantics`.
///
/// The wide helpers are the only integer operations this crate evaluates, and they are reached only
/// from the LLVM backend, so they conform under the `llvm` tag rather than one of their own. The
/// relation is that backend's: equal to [`residue`](mavros_int_semantics::residue) wherever the
/// model has an opinion, which for a sum, difference or product is everywhere and for a division is
/// everywhere but a zero divisor and a signed `INT_MIN / -1`.
///
/// The sweep runs at the narrow widths as well as the wide ones, though the backend routes only
/// the wide ones here. A width-generic body is either right at a width or it is not, and one and
/// two limbs are where a limb-count error is visible rather than averaged over.
#[cfg(test)]
mod int_semantics_conformance {
    use mavros_int_semantics::{
        IntBits, IntOp, MAX_BITS, corners, int_bits::HOST_LIMB_BITS, residue,
    };

    /// [`super::__int_mul`] applied to two patterns of the same width.
    ///
    /// The buffers are the pattern's own limbs, which is the caller's job in the backend: an
    /// `iN` is stored into a slot of `ceil(N / 64)` limbs and the answer is read back out of one.
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

    /// The inputs the model leaves unspecified still get an answer, and it is the VM's.
    ///
    /// A zero divisor answers zero for both quotient and remainder, and a signed `INT_MIN / -1`
    /// wraps to `INT_MIN` with a zero remainder. The guard IR keeps both from any division, so this
    /// is not conformance; it is the helper being total rather than trapping, which is what the
    /// wasm engine would otherwise make of a Rust division by zero.
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

    /// The one input shape where Knuth D's first estimate survives its correction loop and is still
    /// one too large, so the window goes negative and the divisor is added back.
    ///
    /// Hacker's Delight's `divmnu` vector, with its 32-bit digits scaled to 64-bit limbs:
    /// `2^191 + 3` divided by `2^189 + 1`. Normalising shifts both left by two, the estimate from
    /// the top two limbs is 4, and the true quotient is 3. The corner sweep reaches the add-back
    /// too, but only through whichever corners the shared set happens to hold; this pins it to an
    /// input whose path is known.
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
            // Both sides of a multiply are values, so the two lists are the same; taking it from
            // `wide_operands` is what ensures this remains a fact about the _operation_ rather than
            // about this loop, and builds each 256-limb pattern once instead of once per left
            // operand.
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

        // And without this the wide half could contribute nothing at all while the narrow half
        // carried the count on its own.
        assert!(
            at_the_widest > 100,
            "the widest width contributed only {at_the_widest} points"
        );
    }

    /// The truncation is the operation's, not an accident of the buffer being full.
    ///
    /// At a width that is not a limb multiple the product's top limb carries bits the answer does
    /// not, and dropping them is [`IntBits`]'s normalisation rather than anything `__int_mul` does.
    /// This pins the pair that would otherwise only be swept: a product whose true value overflows
    /// the width, at a width that half-fills its top limb.
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
}
