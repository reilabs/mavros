//! A collection of miscellaneous utils for the compiler that don't necessarily have a good place.

use crate::compiler::ssa::hlssa::MAX_SUPPORTED_UNSIGNED_BITS;

pub use mavros_int_semantics::mask as bit_mask;

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

/// The host word an integer pattern carries, panicking above [`MAX_BITS`].
///
/// Its **production** callers are enumerated here and nowhere else (tests reach for it too, to read
/// an answer back as a number):
///
/// - the `Int <-> Field` conversions (`hlssa_to_r1cs`'s `expect_in_u128` included, which comes the
///   other way), which need a limbs-to-field path and so belong with the field-agnosticism work
///   rather than here;
/// - `spread_bits` / `unspread_bits`, whose own signatures are `u128`, and which Phase 5
///   generalises along with the VM's fixed-width spread opcodes;
/// - one genuine host-word arithmetic site, `instrumenter`'s `Radix::Dyn`, which uses the value as
///   a `u128` divisor to build the digits of a radix decomposition.
///
/// Do not add a caller that could ask its question of the pattern instead: [`IntBits`] answers
/// `is_zero` / `is_one` / `is_all_ones`, the bitwise and shift operations, `cast` / `bit_range` and
/// the whole signed reading width-generically, and `usize::try_from` reads an index without the
/// truncation an `as` cast would hide.
///
/// The width cap is 128, so the panic here is unreachable rather than merely unlikely.
///
/// [`IntBits`]: mavros_int_semantics::IntBits
/// [`MAX_BITS`]: mavros_int_semantics::MAX_BITS
///
/// TODO Remove by the end of the `big-int-model` branch work
#[track_caller]
pub fn host_word(pattern: &mavros_int_semantics::IntBits) -> u128 {
    u128::try_from(pattern)
        .unwrap_or_else(|e| panic!("ICE: an integer constant is wider than the host: {e}"))
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

pub fn unspread_bits(v: u128, bits: usize) -> (u128, u128) {
    assert!(
        bits <= MAX_SUPPORTED_UNSIGNED_BITS && bits % 2 == 0,
        "unspread_bits expects an even width up to {MAX_SUPPORTED_UNSIGNED_BITS}, got {bits}"
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

/// Extract odd/even bit streams from a 64-bit spread value.
pub fn unspread_u64(v: u64) -> (u32, u32) {
    let (odd, even) = unspread_bits(v as u128, 64);
    (odd as u32, even as u32)
}

/// Compute spread of a 32-bit value: interleave zero bits between each bit.
pub fn spread_u64(v: u32) -> u64 {
    spread_bits(v as u128, 32) as u64
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
