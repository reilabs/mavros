//! A collection of miscellaneous utils for the compiler that don't necessarily have a good place.

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{IntBits, int_bits::HOST_WORD_BITS};

use crate::compiler::Field;

/// The low-`bits` mask, as a host word.
///
/// A guarded wrapper over [`mavros_int_semantics::mask`], which **saturates** above
/// [`HOST_WORD_BITS`]: the wrong answer here, where every caller passes the result to
/// `HLEmitter::int_const` at `bits`.
#[track_caller]
pub fn bit_mask(bits: usize) -> u128 {
    assert!(
        bits <= HOST_WORD_BITS,
        "a u{HOST_WORD_BITS} mask cannot represent u{bits}"
    );
    mavros_int_semantics::mask(bits)
}

// Checks the claim `FIELD_LIMB_BITS` rests on: an element's canonical limbs are 64 bits. Each
// caller passes `canonical_limbs()` to `IntBits::from_field_limbs`, and nothing else checks that
// the two agree; if they differed, every field-sourced constant would be silently recombined at the
// wrong place values.
const _: () = assert!(
    u64::BITS as usize == mavros_int_semantics::int_bits::FIELD_LIMB_BITS,
    "a canonical field limb is no longer FIELD_LIMB_BITS wide"
);

/// Bit `index` of the little-endian integer `limbs`, zero past its last limb.
#[must_use]
pub fn limb_bit(limbs: &[u64], index: usize) -> bool {
    limbs
        .get(index / u64::BITS as usize)
        .is_some_and(|limb| (limb >> (index % u64::BITS as usize)) & 1 == 1)
}

/// The `width` bits of the little-endian integer `limbs` from bit `offset` up, as little-endian
/// limbs.
#[must_use]
pub fn limb_bit_window(limbs: &[u64], offset: usize, width: usize) -> Vec<u64> {
    let mut window = vec![0u64; width.div_ceil(u64::BITS as usize)];
    for bit in (0..width).filter(|&bit| limb_bit(limbs, offset + bit)) {
        window[bit / u64::BITS as usize] |= 1 << (bit % u64::BITS as usize);
    }
    window
}

/// The host word an integer pattern carries, panicking above the host word.
///
/// Its **production** callers are listed here and nowhere else (tests also use it to read an
/// answer back as a number):
///
/// - `hlssa_to_r1cs`'s `const_u128` chain, which reads a field element back as a small number (a
///   table size, an index, a bit). The host word is what those callers return, not a bound on any
///   integer type, so refusing past it is the honest answer;
/// - `instrumenter`'s `Radix::Dyn`, the one real host-word arithmetic site, which uses the value as
///   a `u128` divisor to build the digits of a radix decomposition.
///
/// Do not add a caller that could ask the pattern instead: [`IntBits`] answers `is_zero` /
/// `is_one` / `is_all_ones`, the bitwise and shift operations, `cast` / `bit_range` and the whole
/// signed reading at any width; [`field_residue`] answers `Int -> Field` at any width; and
/// `usize::try_from` reads an index without the truncation an `as` cast would hide.
#[track_caller]
pub fn host_word(pattern: &IntBits) -> u128 {
    u128::try_from(pattern)
        .unwrap_or_else(|e| ice!("an integer constant is wider than the host: {e}"))
}

/// The field element equal to an integer pattern, or `None` if the specified field cannot contain
/// the integer in question.
///
/// It inverts [`IntBits::from_field_limbs`] in one direction only. That function truncates to the
/// width asked for, so element to pattern and back returns the starting element only where the
/// width holds it; pattern to element and back is exact wherever this answers.
///
/// The bound is on the **value**, not the width: a pattern at or above the modulus has no element
/// equal to it however narrow its type, and every pattern below it has one however wide. Refusing
/// keeps a residue from being minted in place of the value, which nothing downstream could tell
/// apart. A caller that cannot handle a refusal must bound its inputs itself.
#[must_use]
pub fn field_constant(field: FieldConfig, pattern: &IntBits) -> Option<Field> {
    field.from_canonical_limbs(pattern.limbs())
}

/// The unsigned integer whose little-endian 64-bit limbs are `limbs`, such as an element's
/// [`Field::canonical_limbs`] or the field's [`FieldConfig::modulus_limbs`].
#[must_use]
pub fn biguint_of_limbs(limbs: &[u64]) -> num_bigint::BigUint {
    let bytes_le: Vec<u8> = limbs.iter().flat_map(|limb| limb.to_le_bytes()).collect();
    num_bigint::BigUint::from_bytes_le(&bytes_le)
}

/// The `Int -> Field` cast of an integer pattern: its unsigned reading reduced modulo the field's
/// modulus, [`mavros_int_semantics::int_to_field`].
///
/// It agrees with [`field_constant`] wherever that answers, and differs only by also answering a
/// pattern at or above the modulus. A caller that must not mint a residue in place of a value
/// wants [`field_constant`]. This is total.
#[must_use]
pub fn field_residue(field: FieldConfig, pattern: &IntBits) -> Field {
    // Horner over the 64-bit limbs, from the top.
    let base = field.two_pow(64);
    pattern
        .limbs()
        .iter()
        .rev()
        .fold(field.zero(), |acc, &limb| acc * base + field.constant(limb))
}

/// Panic with the canonical ICE for a tuple surviving past the `ElideTuples` pass.
///
/// Everything downstream of `ElideTuples` operates on tuple-free IR; reaching a tuple opcode or
/// tuple type there is a compiler bug. Call this from the (unreachable) tuple arms of downstream
/// passes, analyses, and codegen.
#[track_caller]
pub fn ice_non_elided_tuple() -> ! {
    ice!("Tuple encountered after ElideTuples pass")
}

/// Panic if an `AssertConstant` marker survives its dedicated validation phase.
#[track_caller]
pub fn ice_unvalidated_assert_constant() -> ! {
    ice!("AssertConstant encountered after assert-constant validation")
}

/// Utilities only available in tests.
#[cfg(test)]
pub mod test {
    use mavros_artifacts::FieldConfig;

    use crate::compiler::{
        Field,
        ssa::{ValueId, hlssa::builder::HLEmitter},
    };

    /// Convert the provided `n` into a field value.
    pub fn fr(n: u64) -> Field {
        FieldConfig::bn254().constant(n)
    }

    /// Both configured fields, for a sweep that must hold on each.
    pub const FIELDS: [FieldConfig; 2] = [FieldConfig::bn254(), FieldConfig::goldilocks()];

    /// The field's modulus as a magnitude, the form the reference model takes it in.
    pub fn modulus_of(field: FieldConfig) -> num_bigint::BigUint {
        super::biguint_of_limbs(field.modulus_limbs())
    }

    /// An element's canonical representative.
    pub fn canonical_of(element: &Field) -> num_bigint::BigUint {
        super::biguint_of_limbs(&element.canonical_limbs())
    }

    /// The element whose canonical representative is `canonical`.
    ///
    /// # Panics
    ///
    /// If `canonical` is not below the modulus.
    pub fn element_of(field: FieldConfig, canonical: &num_bigint::BigUint) -> Field {
        field
            .from_canonical_limbs(&canonical.to_u64_digits())
            .unwrap_or_else(|| panic!("{canonical} is not below the modulus"))
    }

    /// The reference model's `Int -> Field` answer, as an element of `field`.
    pub fn model_int_to_field(
        field: FieldConfig,
        pattern: &mavros_int_semantics::IntBits,
    ) -> Field {
        element_of(
            field,
            &mavros_int_semantics::int_to_field(pattern, &modulus_of(field)),
        )
    }

    /// The reference model's `Field -> Int(bits)` answer.
    pub fn model_field_to_int(
        field: FieldConfig,
        element: &Field,
        bits: usize,
    ) -> mavros_int_semantics::IntBits {
        mavros_int_semantics::field_to_int(&canonical_of(element), &modulus_of(field), bits)
    }

    /// `alloc` of a scalar `Ref<Field>` seeded with an inert default value (0).
    ///
    /// The constant is interned (never a block instruction), so the seed never shows up in
    /// `op_counts`; tests that care about the contents `store` to the cell afterward (the store
    /// overwrites the seed).
    pub fn falloc(e: &mut impl HLEmitter) -> ValueId {
        let init = e.field_const(fr(0));
        e.alloc(init)
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use mavros_int_semantics::IntBits;

    use super::*;

    // Eight sites mint a constant with a host-word shift or mask, each guarded by an assert on the
    // host word (or the narrow threshold), not the integer type cap, which is far above both. The
    // tests below pin what those guards rest on:
    //
    //   1. `1u128 << n` panics in debug and wraps in release for `n >= 128`, and an in-range shift
    //      truncates silently when its result does not fit.
    //   2. `mask(n)` above the host word saturates rather than refusing, so a constant minted from
    //      it at a wider width is silently wrong.

    /// Fact 1, at the first width the raised cap admits and the guards refuse.
    #[test]
    fn a_host_word_shift_past_128_has_no_answer() {
        assert!(1u128.checked_shl(128).is_none());
        assert!(1u128.checked_shl(u128::BITS).is_none());
    }

    /// Fact 1's second half, which only the `offset + width <= bits` bound catches.
    ///
    /// The fragile case: `width` and `offset` can each be legal while the mask they build together
    /// is not, and nothing about a `<<` says so.
    #[test]
    fn a_bit_range_mask_truncates_silently_when_only_its_pieces_are_in_range() {
        // Bits 100..200 of a 200-bit value: legal at the raised cap, and neither shift is out of
        // range by itself.
        let (offset, width) = (100usize, 100usize);
        let host = ((1u128 << width) - 1) << offset;

        // What the mask should have been, and what a host word actually answered.
        let intended = ((num_bigint::BigUint::from(1u8) << width) - 1u8) << offset;
        assert_ne!(num_bigint::BigUint::from(host), intended);

        // And the bound that refuses it, stated the way `bit_mask` states it.
        assert!(offset + width > HOST_WORD_BITS);
    }

    /// Fact 2: why `bit_mask` is a guarded wrapper rather than a bare re-export.
    #[test]
    fn the_model_mask_saturates_where_bit_mask_refuses() {
        // At the host word the two agree, so the difference is invisible below it.
        assert_eq!(bit_mask(128), u128::MAX);

        // One bit further the model still answers, but the constant it mints is not all ones:
        // bit 128 is zero.
        let saturated = mavros_int_semantics::mask(129);
        assert_eq!(saturated, u128::MAX);
        let minus_one = IntBits::from_u128(129, saturated);
        assert!(!minus_one.is_all_ones());
        assert_eq!(minus_one.bit(128), Some(false));
    }

    #[test]
    #[should_panic(expected = "cannot represent u129")]
    fn bit_mask_refuses_a_width_past_the_host_word() {
        let _ = bit_mask(129);
    }

    // THE LIMBS-TO-FIELD PATH
    // --------------------------------------------------------------------------------------

    /// Below the host word the reading matches Noir; pinned so the artifacts do not move.
    #[test]
    fn a_narrow_pattern_reads_the_value_a_host_word_did() {
        let field = FieldConfig::bn254();

        for bits in [1usize, 8, 63, 64, 65, 127, 128] {
            for value in [
                0u128,
                1,
                2,
                u128::from(u64::MAX),
                mavros_int_semantics::mask(bits),
            ] {
                let pattern = IntBits::from_u128(bits, value);
                assert_eq!(
                    field_constant(field, &pattern),
                    Some(field.constant(host_word(&pattern))),
                    "u{bits} {value:#x}"
                );
            }
        }
    }

    /// The bound is the value's, so a pattern far past every host word still reads.
    #[test]
    fn a_wide_pattern_below_the_modulus_reads_rather_than_being_refused_for_its_width() {
        let field = FieldConfig::bn254();
        let largest = test::modulus_of(field) - 1u8;

        for bits in [129usize, 256, 1000, 16384] {
            let pattern = IntBits::from_biguint(bits, &largest);
            let element = field_constant(field, &pattern)
                .unwrap_or_else(|| panic!("u{bits} holding p-1 has an element"));

            // Read back through the canonical decomposition `Field -> Int` uses, so the round
            // trip pins the place values, not just the acceptance.
            assert_eq!(
                IntBits::from_field_limbs(&element.canonical_limbs(), bits),
                pattern,
                "u{bits}"
            );
        }
    }

    /// The rule as a relation rather than as a set of examples.
    #[test]
    fn the_reading_is_the_models_magnitude_and_the_bound_is_the_modulus() {
        let field = FieldConfig::bn254();
        let p = test::modulus_of(field);

        let mut answered = 0usize;
        for bits in [1usize, 64, 128, 129, 253, 254, 1000] {
            for pattern in [
                IntBits::zero(bits),
                IntBits::from_u128(bits, 1),
                IntBits::all_ones(bits),
                IntBits::from_biguint(bits, &(p.clone() - 1u8)),
                IntBits::from_biguint(bits, &p),
            ] {
                let magnitude = num_bigint::BigUint::from(&pattern);
                match field_constant(field, &pattern) {
                    Some(element) => {
                        assert!(
                            magnitude < p,
                            "u{bits} {magnitude} is not below the modulus"
                        );
                        assert_eq!(
                            num_bigint::BigUint::from(&IntBits::from_field_limbs(
                                &element.canonical_limbs(),
                                bits.max(254)
                            )),
                            magnitude,
                            "u{bits} did not carry its magnitude across"
                        );
                        answered += 1;
                    }
                    None => assert!(
                        magnitude >= p,
                        "u{bits} {magnitude} declined below the modulus"
                    ),
                }
            }
        }
        assert!(
            answered > 10,
            "only {answered} patterns were carried across"
        );
    }

    /// At the modulus the reduction would give **zero**, and a residue cannot be told apart from
    /// the value it should have been.
    #[test]
    fn a_pattern_at_or_above_the_modulus_declines_rather_than_reducing() {
        let field = FieldConfig::bn254();
        let p = test::modulus_of(field);

        for offset in [0u8, 1] {
            let pattern = IntBits::from_biguint(256, &(p.clone() + offset));
            assert_eq!(field_constant(field, &pattern), None, "p + {offset}");
        }

        // The other half of the decline: a magnitude in limbs past those the field has.
        assert_eq!(field_constant(field, &IntBits::all_ones(16384)), None);
        assert_eq!(field_constant(field, &IntBits::all_ones(256)), None);
    }

    /// The residue is the model's on both fields at every corner, and is [`field_constant`]'s
    /// answer wherever that answers.
    #[test]
    fn the_residue_is_the_models_and_agrees_with_the_exact_reading() {
        use mavros_int_semantics::corners;

        use super::test::{FIELDS, model_int_to_field, modulus_of};

        for field in FIELDS {
            let p = modulus_of(field);
            let mut reduced = 0usize;
            for bits in corners::field_cast_widths(field.field_bit_size() as usize) {
                for pattern in corners::field_cast_patterns(bits, &p) {
                    let residue = field_residue(field, &pattern);
                    assert_eq!(
                        residue,
                        model_int_to_field(field, &pattern),
                        "{field:?}: {pattern:?}"
                    );
                    match field_constant(field, &pattern) {
                        Some(exact) => assert_eq!(exact, residue, "{field:?}: {pattern:?}"),
                        None => reduced += 1,
                    }
                }
            }
            assert!(reduced > 20, "only {reduced} corners on {field:?} reduce");
        }
    }
}
