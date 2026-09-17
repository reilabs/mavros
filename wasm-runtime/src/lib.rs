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
// ABI, following the convention above: the result comes back through a pointer,
// and the operands are pointers too because a width-generic helper cannot take
// an `iN` by value — one body serves every width, so the width arrives as the
// limb count `limbs` instead of in the type.
//
//   __int_mul(result_ptr, a_ptr, b_ptr, limbs)
//
// Every buffer is `limbs` little-endian `u64`s, which is exactly how LLVM lays
// an `iN` out in memory on this little-endian target: the caller stores an
// `i64*limbs` value into the slot and loads the answer back out of it.
//
// LLVM's own expansion of a wide `mul` is straight-line code quadratic in the
// width -- 7.2 MB at `i16384`, and past the wasm engine's function-size and
// local-count caps from about `i5700` up -- where a loop over the limbs is one
// small function serving every width.
// ═══════════════════════════════════════════════════════════════════════════════

/// The low `limbs` limbs of `a * b`.
///
/// Schoolbook, dropping the columns at or above `limbs` rather than computing and discarding
/// them: the answer is taken modulo `2^(64*limbs)` and those columns cannot reach it.
///
/// The column accumulator needs no wrapping arithmetic. A limb product plus the running cell plus
/// the carry is at most `(2^64 - 1)^2 + 2 * (2^64 - 1)`, which is `2^128 - 1` exactly, so a `u128`
/// holds every column.
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
        result.write_bytes(0, k);

        for i in 0..k {
            let ai = u128::from(*a.add(i));
            let mut carry = 0u64;
            for j in 0..(k - i) {
                let column =
                    ai * u128::from(*b.add(j)) + u128::from(*result.add(i + j)) + u128::from(carry);
                *result.add(i + j) = column as u64;
                carry = (column >> 64) as u64;
            }
        }
    }
}

/// The runtime helper's conformance relation to the normative model in `mavros-int-semantics`.
///
/// [`__int_mul`] is the one integer operation this crate evaluates, and it is reached only from
/// the LLVM backend, so it conforms under the `llvm` tag rather than one of its own. The relation
/// is that backend's: equal to [`residue`](mavros_int_semantics::residue) wherever the model has
/// an opinion, which for a multiply is everywhere.
///
/// The sweep runs at the narrow widths as well as the wide ones, though the backend routes only
/// the wide ones here. A width-generic body is either right at a width or it is not, and one and
/// two limbs are where a limb-count error is visible rather than averaged over.
#[cfg(test)]
mod int_semantics_conformance {
    use mavros_int_semantics::{IntBits, IntOp, corners, residue};

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

    #[test]
    fn the_helper_multiply_agrees_with_the_model() {
        let mut checked = 0usize;
        let mut at_the_widest = 0usize;

        // Every wide width the backend can route here, plus the narrow ones it never will.
        let mut widths = vec![1usize, 2, 7, 8, 63, 64, 65, 96, 127, 128];
        widths.extend(corners::wide_widths_for(false));

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
        // carried the count on its own. A sweep whose wide widths come from a filtered set is an
        // ordinary shape here, not a hypothetical: `corners::wide_widths_for(true)` is empty.
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
