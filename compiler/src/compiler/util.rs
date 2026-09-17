//! A collection of miscellaneous utils for the compiler that don't necessarily have a good place.

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{IntBits, int_bits::HOST_WORD_BITS};

use crate::compiler::Field;

/// The low-`bits` mask, as a host word.
///
/// A guarded wrapper over [`mavros_int_semantics::mask`]. The model's own `mask` **saturates**
/// above [`HOST_WORD_BITS`], which is the wrong answer here where every caller feeds the result to
/// `HLEmitter::int_const` at `bits`.
#[track_caller]
pub fn bit_mask(bits: usize) -> u128 {
    assert!(
        bits <= HOST_WORD_BITS,
        "a u{HOST_WORD_BITS} mask cannot represent u{bits}"
    );
    mavros_int_semantics::mask(bits)
}

// The claim `FIELD_LIMB_BITS` is written on, checked rather than stated in prose.
//
// `ark_ff::BigInt<N>` is `[u64; N]`, so a canonical field representation's limb _count_ varies from
// field to field while its limb _width_ does not. The limb _type_ is already pinned at every call
// site, because `IntBits::from_field_limbs` takes a `&[u64]` and each caller hands it
// `into_bigint().0`. What nothing else checks is that the model's 64 and arkworks' 64 are the same
// number: if they ever parted, every field-sourced constant would be recombined with the wrong
// place values and nothing would say so.
//
// It lives here because this is the lowest crate where both are in scope: `mavros-int-semantics`
// cannot see `BigInt`, and must not, being the crate every evaluator is measured against.
const _: () = assert!(
    size_of::<ark_ff::BigInt<1>>() * 8 == mavros_int_semantics::int_bits::FIELD_LIMB_BITS,
    "a canonical field limb is no longer FIELD_LIMB_BITS wide"
);

/// The host word an integer pattern carries, panicking above the host word.
///
/// Its **production** callers are enumerated here and nowhere else (tests reach for it too, to read
/// an answer back as a number):
///
/// - `hlssa_to_r1cs`'s `const_u128` chain, which reads a field element back as a small number —
///   a table size, an index, a bit. The host word there is a fact about what those callers return
///   rather than a bound on any integer type, and the refusal past it is the honest answer;
/// - `spread_bits` / `unspread_bits`, whose own signatures are `u128`, and which Phase 5
///   generalises along with the VM's fixed-width spread opcodes;
/// - one genuine host-word arithmetic site, `instrumenter`'s `Radix::Dyn`, which uses the value as
///   a `u128` divisor to build the digits of a radix decomposition.
///
/// Do not add a caller that could ask its question of the pattern instead: [`IntBits`] answers
/// `is_zero` / `is_one` / `is_all_ones`, the bitwise and shift operations, `cast` / `bit_range` and
/// the whole signed reading width-generically; [`field_constant`] answers the `Int -> Field`
/// direction at the field's own bound; and `usize::try_from` reads an index without the truncation
/// an `as` cast would hide.
///
/// TODO Remove by the end of the `big-int-model` branch work
#[track_caller]
pub fn host_word(pattern: &IntBits) -> u128 {
    u128::try_from(pattern)
        .unwrap_or_else(|e| panic!("ICE: an integer constant is wider than the host: {e}"))
}

/// The field element an integer pattern denotes, or `None` if this field cannot carry it.
///
/// The limbs-to-field half of the `Int <-> Field` conversions. The value it carries across is the
/// model's: a cast is a raw-bits conversion, so a pattern denotes the magnitude its bits spell.
///
/// It is a **left** inverse of [`IntBits::from_field_limbs`] rather than an inverse: that direction
/// truncates to the width asked for, so the round trip returns the element it started from only
/// where the width was wide enough to hold it. In the other order the round trip is exact wherever
/// this answers at all.
///
/// The bound is on the **value** and not on the width: a pattern at or above the modulus has no
/// element however narrow its type, and every pattern below one has an element however wide. The
/// refusal is what keeps a residue from being minted in place of the value, which nothing
/// downstream could tell apart from the value itself. A caller that cannot handle a refusal here
/// must bound its own inputs.
#[must_use]
pub fn field_constant(field: FieldConfig, pattern: &IntBits) -> Option<Field> {
    // Four limbs, spelled as a literal because that is what it is: `FieldConfig::from_bigint`
    // takes an `ark_ff::BigInt<4>`, so the count is fixed by the signature rather than asked of the
    // field. Reading it off `modulus_limbs` would look field-derived without being so.
    // FIELD-ASSUMPTION: L3-felt-limbs
    let mut canonical = [0u64; 4];

    let limbs = pattern.limbs();
    if limbs.len() > canonical.len() && limbs[canonical.len()..].iter().any(|&limb| limb != 0) {
        return None;
    }
    for (slot, limb) in canonical.iter_mut().zip(limbs) {
        *slot = *limb;
    }

    field.from_bigint(ark_ff::BigInt::new(canonical))
}

/// Panic with the canonical ICE for a tuple surviving past the `ElideTuples` pass.
///
/// Everything downstream of `ElideTuples` operates on tuple-free IR; reaching a tuple opcode or
/// tuple type there is a compiler bug. Call this from the (unreachable) tuple arms of downstream
/// passes, analyses, and codegen.
#[track_caller]
pub fn ice_non_elided_tuple() -> ! {
    panic!("ICE: Tuple encountered after ElideTuples pass")
}

/// Panic if an `AssertConstant` marker survives its dedicated validation phase.
#[track_caller]
pub fn ice_unvalidated_assert_constant() -> ! {
    panic!("ICE: AssertConstant encountered after assert-constant validation")
}

pub fn spread_bits(v: u128, bits: usize) -> u128 {
    assert!(
        bits <= 64,
        "spread_bits only supports widths up to 64, got {bits}"
    );

    let mut x = v;
    x = (x | (x << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333u128;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555u128;
    x
}

/// The widest spread [`unspread_bits`] can read back, and so the widest `Unspread` any evaluator
/// may be asked for.
pub const UNSPREAD_INPUT_MAX: usize = HOST_WORD_BITS;

pub fn unspread_bits(v: u128, bits: usize) -> (u128, u128) {
    assert!(
        bits <= UNSPREAD_INPUT_MAX && bits % 2 == 0,
        "unspread_bits expects an even width up to {UNSPREAD_INPUT_MAX}, got {bits}"
    );

    fn compact_bits(mut x: u128) -> u128 {
        x &= 0x5555_5555_5555_5555_5555_5555_5555_5555u128;
        x = (x | (x >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333u128;
        x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128;
        x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128;
        x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128;
        x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128;
        x = (x | (x >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFFu128;
        x
    }

    let even = compact_bits(v);
    let odd = compact_bits(v >> 1);
    (odd, even)
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

    // Eight sites create a constant with a host-word shift or mask, and each is guarded by an
    // assert naming the host word (or the narrow threshold) rather than the integer type _cap_: the
    // cap is far above both, so a guard written against it would admit a silent overflow instead of
    // refusing. We ensure the arithmetic facts these guards rest on (for the moment) below:
    //
    //   1. `1u128 << n` is a debug panic and a release wraparound for every `n >= 128`; and a shift
    //      that stays in range still **truncates silently** once its result would not fit.
    //   2. `mask(n)` above the host word **saturates** rather than refusing, so a constant minted
    //      from it at a wider width is a wrong value with nothing to say so.

    /// Fact 1, at the first width the raised cap admits and the guards refuse.
    #[test]
    fn a_host_word_shift_past_128_has_no_answer() {
        assert!(1u128.checked_shl(128).is_none());
        assert!(1u128.checked_shl(u128::BITS).is_none());
    }

    /// Fact 1's second half, which the `offset + width <= bits` bound is what actually catches.
    ///
    /// This is the fragile one: `width` and `offset` can each be perfectly legal on their own while
    /// the mask they build together is not, and nothing about a `<<` says so.
    #[test]
    fn a_bit_range_mask_truncates_silently_when_only_its_pieces_are_in_range() {
        // Bits 100..200 of a 200-bit value: legal at the raised cap, and neither shift is out of
        // range on its own.
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
        // At the host word the two agree, which is what makes the difference invisible below it.
        assert_eq!(bit_mask(128), u128::MAX);

        // One bit further the model still answers -- and the constant that answer mints is not the
        // all-ones pattern the caller asked for, because bit 128 is zero.
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

    /// The modulus as a magnitude, which is the number `field_constant` declines at.
    fn modulus(field: FieldConfig) -> num_bigint::BigUint {
        field
            .modulus_limbs()
            .iter()
            .rev()
            .fold(num_bigint::BigUint::ZERO, |acc, &limb| {
                (acc << 64) | num_bigint::BigUint::from(limb)
            })
    }

    /// Below the host word the reading is as with Noir, so we pin it to ensure artifact neutrality.
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
        let largest = modulus(field) - 1u8;

        for bits in [129usize, 256, 1000, 16384] {
            let pattern = IntBits::from_biguint(bits, &largest);
            let element = field_constant(field, &pattern)
                .unwrap_or_else(|| panic!("u{bits} holding p-1 has an element"));

            // Read back through the same canonical decomposition the `Field -> Int` direction uses,
            // so the round trip pins the place values and not just the acceptance.
            assert_eq!(
                IntBits::from_field_limbs(&element.into_bigint().0, bits),
                pattern,
                "u{bits}"
            );
        }
    }

    /// The rule as a relation rather than as a set of examples.
    #[test]
    fn the_reading_is_the_models_magnitude_and_the_bound_is_the_modulus() {
        let field = FieldConfig::bn254();
        let p = modulus(field);

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
                                &element.into_bigint().0,
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

    /// At the modulus the reduction would answer **zero**, and a residue is indistinguishable from
    /// the value it should have been.
    #[test]
    fn a_pattern_at_or_above_the_modulus_declines_rather_than_reducing() {
        let field = FieldConfig::bn254();
        let p = modulus(field);

        for offset in [0u8, 1] {
            let pattern = IntBits::from_biguint(256, &(p.clone() + offset));
            assert_eq!(field_constant(field, &pattern), None, "p + {offset}");
        }

        // And a pattern whose magnitude is in limbs the field has none of, which is the other half
        // of the decline.
        assert_eq!(field_constant(field, &IntBits::all_ones(16384)), None);
        assert_eq!(field_constant(field, &IntBits::all_ones(256)), None);
    }
}
