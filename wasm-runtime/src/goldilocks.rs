//! The Goldilocks field's runtime entry points, under the symbol names `FieldRuntimeSymbols`
//! (`compiler/src/compiler/codegen/llssa_to_llvm.rs`) gives that field.
//!
//! An element is one Montgomery-form cell. LLVM returns a one-`i64` struct directly on wasm32 and
//! passes each operand as one `i64`, so an entry point takes and returns `u64`s, with no result
//! pointer: `__goldilocks_mul(a, b) -> result`.

use mavros_limb_arith::goldilocks as arith;

#[unsafe(no_mangle)]
pub extern "C" fn __goldilocks_mul(a: u64, b: u64) -> u64 {
    arith::mul(a, b)
}

/// A one-limb integer to Montgomery storage, reduced modulo the modulus.
#[unsafe(no_mangle)]
pub extern "C" fn __goldilocks_from_limbs(value: u64) -> u64 {
    arith::to_montgomery(arith::reduce64(value))
}

/// Montgomery storage to its canonical limb.
#[unsafe(no_mangle)]
pub extern "C" fn __goldilocks_to_limbs(cell: u64) -> u64 {
    arith::from_montgomery(cell)
}

#[unsafe(no_mangle)]
pub extern "C" fn __goldilocks_sub(a: u64, b: u64) -> u64 {
    arith::sub(a, b)
}

#[unsafe(no_mangle)]
pub extern "C" fn __goldilocks_add(a: u64, b: u64) -> u64 {
    arith::add(a, b)
}

/// `a / b`, and zero when `b` is zero.
#[unsafe(no_mangle)]
pub extern "C" fn __goldilocks_div(a: u64, b: u64) -> u64 {
    match arith::inverse(b) {
        Some(inverse) => arith::mul(a, inverse),
        None => 0,
    }
}

/// Whether `a < b` as canonical integers.
#[unsafe(no_mangle)]
pub extern "C" fn __goldilocks_lt(a: u64, b: u64) -> bool {
    arith::from_montgomery(a) < arith::from_montgomery(b)
}

/// The entry points against `u128` arithmetic modulo `p`, through the storage form codegen calls
/// them with.
#[cfg(test)]
mod tests {
    use super::*;
    use arith::{EPSILON, MODULUS};
    use mavros_int_semantics::{IntBits, corners, field_to_int, int_bits::HOST_LIMB_BITS};
    use num_bigint::BigUint;

    const P: u128 = MODULUS as u128;

    /// `x·2^64 mod p`.
    fn storage(canonical: u64) -> u64 {
        ((u128::from(canonical) << 64) % P) as u64
    }

    fn samples() -> [u64; 11] {
        [
            0,
            1,
            2,
            7,
            EPSILON,
            EPSILON + 1,
            1 << 32,
            1 << 63,
            (MODULUS - 1) / 2,
            MODULUS - 2,
            MODULUS - 1,
        ]
    }

    #[test]
    fn arithmetic_matches_the_field() {
        for a in samples() {
            for b in samples() {
                let (wide_a, wide_b) = (u128::from(a), u128::from(b));
                let (x, y) = (storage(a), storage(b));
                assert_eq!(
                    __goldilocks_add(x, y),
                    storage(((wide_a + wide_b) % P) as u64),
                    "{a} + {b}"
                );
                assert_eq!(
                    __goldilocks_sub(x, y),
                    storage(((wide_a + P - wide_b) % P) as u64),
                    "{a} - {b}"
                );
                assert_eq!(
                    __goldilocks_mul(x, y),
                    storage(((wide_a * wide_b) % P) as u64),
                    "{a} * {b}"
                );
                if b == 0 {
                    assert_eq!(__goldilocks_div(x, y), 0, "{a} / 0");
                } else {
                    assert_eq!(__goldilocks_mul(__goldilocks_div(x, y), y), x, "{a} / {b}");
                }
            }
        }
    }

    /// Ordered on the canonical integers. The samples include a pair whose Montgomery storage
    /// orders the other way, so comparing storage fails this.
    #[test]
    fn less_than_compares_canonical_integers() {
        let mut storage_disagrees = false;
        for a in samples() {
            for b in samples() {
                let lt = __goldilocks_lt(storage(a), storage(b));
                assert_eq!(lt, a < b, "{a} < {b}");
                storage_disagrees |= (a < b) != (storage(a) < storage(b));
            }
        }
        assert!(storage_disagrees);
    }

    #[test]
    fn limbs_are_the_canonical_integer() {
        for x in samples() {
            assert_eq!(__goldilocks_to_limbs(storage(x)), x);
            assert_eq!(__goldilocks_from_limbs(x), storage(x));
        }
        // A limb past the modulus reduces.
        assert_eq!(__goldilocks_from_limbs(MODULUS), 0);
        assert_eq!(
            __goldilocks_from_limbs(u64::MAX),
            storage(u64::MAX - MODULUS)
        );
    }

    /// The limb conversions, which both field casts lower through, answer the model at every corner
    /// the one-limb packing holds: `from_limbs` reduces, and `to_limbs` gives the canonical value,
    /// which the lowering then truncates or zero-extends.
    #[test]
    fn the_goldilocks_field_conversions_agree_with_the_model() {
        let modulus = BigUint::from(MODULUS);
        let low_limb = |value: &BigUint| value.iter_u64_digits().next().unwrap_or(0);

        let mut checked = 0usize;
        for bits in corners::field_cast_widths(64) {
            if bits > HOST_LIMB_BITS {
                continue;
            }
            for pattern in corners::field_cast_patterns(bits, &modulus) {
                let reduced = mavros_int_semantics::int_to_field(&pattern, &modulus);
                assert_eq!(
                    __goldilocks_from_limbs(pattern.limbs()[0]),
                    storage(low_limb(&reduced)),
                    "__goldilocks_from_limbs of {pattern:?}"
                );
                checked += 1;
            }
        }
        for canonical in corners::field_cast_elements(&modulus) {
            let limb = __goldilocks_to_limbs(storage(low_limb(&canonical)));
            assert_eq!(
                IntBits::from_limbs(HOST_LIMB_BITS, &[limb]),
                field_to_int(&canonical, &modulus, HOST_LIMB_BITS),
                "__goldilocks_to_limbs of {canonical}"
            );
            checked += 1;
        }
        assert!(checked > 50, "only {checked} conversions were checked");
    }
}
