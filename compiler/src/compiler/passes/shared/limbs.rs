//! The witness limb: how wide a piece of an integer can be for a given field.
//!
//! This governs witness layout and the widths the spread-based bitwise lowering works at. It is
//! also the width the carry chain and the schoolbook product in `passes::wide_witness_ints` work
//! at — see `docs/field-agnosticism.md`, Layer 6.

use num_bigint::BigInt;
use num_traits::One;

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::int_bits::{HOST_LIMB_BITS, HOST_WORD_BITS};

use crate::compiler::{
    analysis::value_range_analysis::field_modulus,
    passes::shared::unsupported::unsupported_on_this_field,
    ssa::{
        ValueId,
        hlssa::{CastTarget, MAX_POW2_TABLE_SIZE, builder::HLEmitter},
    },
};

// THE NARROW/WIDE DISPATCH THRESHOLD
// ================================================================================================

/// The host half of [`narrow_int_bits`]: the widest value the narrow lowerings can _mint_, since
/// they build constants through `HLEmitter::int_const` (a `u128` payload) and host-word shifts.
const NARROW_HOST_BITS: usize = HOST_WORD_BITS;

/// The widest integer the single-cell lowerings handle.
///
/// The smaller of two bounds, each binding on a real field:
///
/// - the **field**: a narrow lowering works in one field element, so `2^bits <= p` must hold for
///   its range check to tell a residue from the true value;
/// - the **host**: its constants are host words ([`NARROW_HOST_BITS`]).
///
/// A **dispatch** threshold, **not a soundness bound**: it picks a lowering's shape and replaces
/// none of the per-operation predicates. Past it, a witnessed operation goes:
///
/// - bitwise: `witness_bitwise` decomposes into half-limbs while the value has an element, and
///   `passes::wide_witness_ints` works limb by limb beyond that;
/// - unsigned sum, difference and ordering: one cell while the field holds the sum, and the carry
///   chain in `passes::wide_witness_ints` beyond it;
/// - unsigned product, and division and remainder (whose check is a product plus the remainder):
///   one cell while [`single_cell_product_fits`] holds, and the schoolbook beyond it;
/// - unsigned shifts: one cell while [`single_cell_shift_fits`] holds, and limb-wise beyond it;
/// - equality: no width limit;
/// - signed sum, difference and ordering: the unsigned route, with the sign bits checked beside it;
/// - signed product, division and remainder: one cell while [`single_cell_signed_product_fits`]
///   holds, and beyond it the unsigned gadgets over the operands' magnitudes;
/// - signed shifts: a left shift is the unsigned one, and a right shift is the unsigned one with
///   the sign filled in;
/// - sign extension: fills the limbs past the element.
///
/// Bit windows and range checks are bounded by the field, not by this.
pub fn narrow_int_bits(field: FieldConfig) -> usize {
    narrow_int_bits_for_modulus(&field_modulus(field))
}

/// The body of [`narrow_int_bits`], stated against the modulus so that a field this compiler cannot
/// yet be configured for can still be checked.
pub fn narrow_int_bits_for_modulus(modulus: &BigInt) -> usize {
    (NARROWEST_LIMB_BITS..=NARROW_HOST_BITS)
        .rev()
        .find(|&bits| combined_limbs_fit_modulus(modulus, bits, 1))
        .unwrap_or(NARROWEST_LIMB_BITS)
}

/// The largest `LookupTarget::Pow2` table size whose widest row `field` can hold.
///
/// A size-`s` table's widest row is `2^(2^s - 1)`. A row at or past the modulus wraps identically
/// in every evaluator, so the table silently stops holding powers of two: a soundness bound, and so
/// derived from the field rather than named.
///
/// Clamped to [`MAX_POW2_TABLE_SIZE`], the backends' table-cache size. bn254 reaches it exactly (7,
/// widest row `2^127`); goldilocks gives 6.
pub fn max_pow2_table_size(field: FieldConfig) -> usize {
    max_pow2_table_size_for_modulus(&field_modulus(field))
}

/// The body of [`max_pow2_table_size`], stated against the modulus so a field this compiler cannot
/// yet be configured for can still be checked.
pub fn max_pow2_table_size_for_modulus(modulus: &BigInt) -> usize {
    (0..=MAX_POW2_TABLE_SIZE)
        .rev()
        .find(|&size| widest_pow2_row(size) < *modulus)
        // Unreachable for any modulus above one (size zero's widest row is `2^0`).
        .unwrap_or(0)
}

/// The value in the last row of a size-`size` powers-of-two table: `2^(2^size - 1)`.
fn widest_pow2_row(size: usize) -> BigInt {
    BigInt::one() << ((1usize << size) - 1)
}

// THE WITNESS LIMB WIDTH
// ================================================================================================

/// The narrowest admissible limb width, and the fallback that keeps [`narrow_int_bits_for_modulus`]
/// total.
const NARROWEST_LIMB_BITS: usize = 1;

/// The admissible limb widths, widest first: the powers of two from [`NARROWEST_LIMB_BITS`] up to
/// [`HOST_LIMB_BITS`], since a limb lives in a host word.
fn candidate_widths() -> Vec<usize> {
    (NARROWEST_LIMB_BITS..=HOST_LIMB_BITS)
        .rev()
        .filter(|w| w.is_power_of_two())
        .collect()
}

/// The witness limb width `h` for `field`.
///
/// The width a value is cut into before its pieces are packed into field elements; see
/// [`limb_bits_for_modulus`]. Each condition below asks for more than a limb product, so each is a
/// separate predicate:
///
/// - that a lowering which scales a limb column by a place value still fits
///   ([`two_limb_product_packing_fits`]);
/// - that the sum of two spreads still fits ([`spread_sum_fits_field`]);
/// - that the limbs recombine into one cell, which [`combine_limbs_of_value`] asks of the width of
///   the value they came from.
pub fn witness_limb_bits(field: FieldConfig) -> usize {
    limb_bits_for_modulus(&field_modulus(field))
}

/// The widest admissible `h` with `2h <= widest_injective_int_bits`: every product of two limbs is
/// then a distinct element of `modulus`.
///
/// So is a limb times a power of two up to `2^h`, and on bn254 and goldilocks the bits between `2h`
/// and the injective width hold the partial products and carries a schoolbook column accumulates.
/// Asking only that a bare `(2^h - 1)^2` stay below the modulus would give goldilocks a 32-bit
/// limb, whose product has 64 bits: past the injective width that the lowerings state their bounds
/// against.
///
/// # Panics
///
/// If `modulus` is below 4, where two bits are not injective.
pub fn limb_bits_for_modulus(modulus: &BigInt) -> usize {
    let widest = widest_injective_int_bits_for_modulus(modulus);
    candidate_widths()
        .into_iter()
        .find(|&limb_bits| 2 * limb_bits <= widest)
        .unwrap_or_else(|| panic!("a field of modulus {modulus} has no witness limb"))
}

/// Half a witness limb: the width the spread-based bitwise lowering works at.
///
/// A `k`-bit value spreads to `2k` bits, so a half-limb spread fills exactly one limb.
///
/// # Panics
///
/// If the limb has no exact half: a decomposition range-checks its derived low half to exactly this
/// many bits, so an odd `h` would leave a bit unconstrained. [`limb_bits_for_modulus`] admits only
/// powers of two.
pub fn witness_half_limb_bits(field: FieldConfig) -> usize {
    let limb_bits = witness_limb_bits(field);
    assert!(
        limb_bits >= 2 && limb_bits.is_power_of_two(),
        "a {limb_bits}-bit limb has no exact half, so two half-limbs would not cover it"
    );
    limb_bits / 2
}

/// Whether a `bits`-wide value's two-limb schoolbook product recombines into one field element.
///
/// `bits` must split into exactly two limbs, and the packed column must fit: [`witness_limb_bits`]
/// certifies only that one `2h`-bit partial product is injective, and
/// `lo·lo + (lo·hi + hi·lo)·2^h` scales a two-product column ([`two_limb_packing_fits_modulus`]).
pub fn two_limb_product_packing_fits(field: FieldConfig, bits: usize) -> bool {
    let limb_bits = witness_limb_bits(field);
    bits.div_ceil(limb_bits) == 2 && two_limb_packing_fits_modulus(&field_modulus(field), limb_bits)
}

/// Whether `lo·lo + (lo·hi + hi·lo)·2^h` stays below `modulus` for limbs of `limb_bits` bits.
///
/// With `M = 2^h − 1` the packed value reaches `M²·(2^{h+1} + 1)`, about `2^{3h+1}` (~2^193 on
/// bn254). The limb width does not imply it: the limb rule asks only that `2h` bits be injective.
pub fn two_limb_packing_fits_modulus(modulus: &BigInt, limb_bits: usize) -> bool {
    let max_limb = (BigInt::one() << limb_bits) - BigInt::one();
    let packed = &max_limb * &max_limb * ((BigInt::one() << (limb_bits + 1)) + BigInt::one());
    packed < *modulus
}

/// Whether the sum of two `bits`-wide spreads still fits one element of `field`.
///
/// The spread-then-add bitwise lowering adds two spreads and hands the sum to `Unspread`, which
/// reads the interleaved bits back out. Once the sum can wrap, `Unspread` reads a residue instead.
pub fn spread_sum_fits_field(bits: usize, field: FieldConfig) -> bool {
    spread_sum_fits_modulus(bits, &field_modulus(field))
}

/// Whether the sum of two `bits`-wide spreads stays below `modulus`.
///
/// A `bits`-wide spread is at most `(4^bits − 1)/3`. Stated against the modulus: a bit-length bound
/// would lose a whole width, refusing a `u32` bitwise op on goldilocks.
pub fn spread_sum_fits_modulus(bits: usize, modulus: &BigInt) -> bool {
    let max_spread = ((BigInt::one() << (2 * bits)) - BigInt::one()) / 3;
    (max_spread << 1) < *modulus
}

/// Whether `limb_count` limbs of `limb_bits` bits each recombine below `modulus`.
///
/// Each limb is below `2^limb_bits`, so the recombination is below `2^(limb_count · limb_bits)`;
/// the bound is tight, since every limb may be all ones.
///
/// This is the question [`narrow_int_bits`] asks of one limb. Recombination does not use it:
/// [`combine_limbs_of_value`] bounds by the width of the value the limbs came from instead, since
/// a top limb narrower than the stride would make this refuse widths that fit.
pub fn combined_limbs_fit_modulus(modulus: &BigInt, limb_bits: usize, limb_count: usize) -> bool {
    (BigInt::one() << (limb_bits * limb_count)) <= *modulus
}

/// The widest integer type whose every value is a distinct element of `field`.
///
/// Up to this width `Int -> Field` is injective; one bit wider, both `0` and `p` are values of the
/// type.
pub fn widest_injective_int_bits(field: FieldConfig) -> usize {
    widest_injective_int_bits_for_modulus(&field_modulus(field))
}

/// The body of [`widest_injective_int_bits`], stated against the modulus so that a field this
/// compiler cannot yet be configured for can still be checked.
pub fn widest_injective_int_bits_for_modulus(modulus: &BigInt) -> usize {
    usize::try_from(modulus.bits())
        .unwrap_or(usize::MAX)
        .saturating_sub(1)
}

/// The widest unsigned integer whose sum or difference `field` holds in one element.
///
/// A range check at `bits` tells an overflowing sum (up to `2^(bits + 1) - 2`) or a borrowing
/// difference (`p - d`) from an honest answer only while `2^(bits + 1) < p`: one bit short of the
/// widest injective width. Past it the carry chain takes the operation.
pub fn widest_cell_sum_bits(field: FieldConfig) -> usize {
    widest_injective_int_bits(field) - 1
}

/// Whether the single-cell multiplier handles a witnessed unsigned product at `bits`.
///
/// An unsigned division or remainder takes the single cell at the same widths, and its
/// `q·d + r` reaches past the bare product `(2^bits − 1)^2`, so both are held to
/// [`single_cell_division_fits_modulus`]. The one exception is the two-limb schoolbook at exactly
/// a double host limb, which has a separate packing predicate.
///
/// Past this, an unsigned product, division or remainder goes through `passes::wide_witness_ints`,
/// which runs first and so is the pass that reads this predicate.
pub fn single_cell_product_fits(field: FieldConfig, bits: usize) -> bool {
    single_cell_division_fits_modulus(&field_modulus(field), bits)
        || (bits == 2 * HOST_LIMB_BITS && two_limb_product_packing_fits(field, bits))
}

/// Whether the single cell takes a witnessed **signed** product, quotient or remainder at `bits`.
///
/// Its gadgets reach less than the unsigned division: the decoded product spans
/// `±2^(2 · bits − 2)`, and the magnitudes it divides are at most `2^(bits − 1)`. The unsigned
/// bound is sufficient for both, and one width short on bn254, where an int127 would fit.
pub fn single_cell_signed_product_fits(field: FieldConfig, bits: usize) -> bool {
    single_cell_division_fits_modulus(&field_modulus(field), bits)
}

/// Whether the single-cell shift takes a witnessed unsigned shift at `bits`, `left` for a `<<`.
///
/// A left shift forms `value · 2^n` with `n` at most `bits − 1` and splits it at `2^bits`, which is
/// unique while `2^(2 · bits − 1)` is injective. A right shift is the single-cell division by
/// `2^n`, which reads its divisor at the full width, so it takes the division's bound.
pub fn single_cell_shift_fits(field: FieldConfig, bits: usize, left: bool) -> bool {
    if left {
        2 * bits - 1 <= widest_injective_int_bits(field)
    } else {
        single_cell_division_fits_modulus(&field_modulus(field), bits)
    }
}

/// Whether the single-cell division `n == q·d + r` at `bits` holds as integers below `modulus`.
///
/// `d` is a `bits`-wide operand, and `q` and the derived `r` are range-checked at `bits` at most,
/// so `q·d + r` reaches `(2^bits − 1)^2 + 2^bits − 1 = 2^bits · (2^bits − 1)`. Below the modulus
/// both sides are canonical, so the identity modulo `p` is one between integers. Stated against the
/// modulus: on goldilocks `2^32 · (2^32 − 1)` is exactly `p − 1`, which a bit-length bound would
/// refuse.
pub fn single_cell_division_fits_modulus(modulus: &BigInt, bits: usize) -> bool {
    let top = BigInt::one() << bits;
    &top * (&top - BigInt::one()) < *modulus
}

/// `ceil(log2(n))`, the number of bits that count `0..n`: the width of a `bits`-wide shift's legal
/// amounts, and the depth of a barrel over `n` limbs.
pub fn ceil_log2(n: usize) -> usize {
    (n - 1).checked_ilog2().map_or(0, |log| log as usize + 1)
}

// LIMB DECOMPOSITIONS
// ================================================================================================

/// A value decomposed into limbs at a uniform place-value stride, least significant limb first.
///
/// `limb_bits` is the **stride**: every limb is `Int(limb_bits)` except a ragged top limb, typed at
/// the bits it carries. Do not infer the top limb's typed width from it.
pub struct WitnessLimbs {
    /// The width of each limb, in bits.
    pub limb_bits: usize,

    /// The limbs, least significant first.
    pub limbs: Vec<ValueId>,
}

impl WitnessLimbs {
    /// The two limbs, least significant first, or a panic if there are not exactly two.
    pub fn pair(&self) -> (ValueId, ValueId) {
        assert_eq!(
            self.limbs.len(),
            2,
            "this lowering reads a decomposition as a pair, but it has {} limbs of {} bits",
            self.limbs.len(),
            self.limb_bits
        );
        (self.limbs[0], self.limbs[1])
    }
}

/// Split `value` into `count` limbs of `limb_bits` bits each, least significant first.
///
/// `count * limb_bits` must not exceed the value's width: a `BitRange` past its source is rejected,
/// so a value that is not a whole number of limbs has to be widened before it can be split.
pub fn split_into_limbs(
    b: &mut impl HLEmitter,
    value: ValueId,
    limb_bits: usize,
    count: usize,
) -> WitnessLimbs {
    let mut limbs = Vec::with_capacity(count);
    for i in 0..count {
        limbs.push(extract_limb(b, value, i * limb_bits, limb_bits));
    }
    WitnessLimbs { limb_bits, limbs }
}

/// Extract the `limb_bits` bits of `value` starting at `offset` as an integer of that width.
pub fn extract_limb(
    b: &mut impl HLEmitter,
    value: ValueId,
    offset: usize,
    limb_bits: usize,
) -> ValueId {
    let limb = b.bit_range(value, offset, limb_bits);
    b.cast_to(CastTarget::Int(limb_bits), limb)
}

/// Recombine `limbs` into one field value, `limb[0] + limb[1] * 2^h + ...`, bounded by the
/// **width of the value they came from** rather than by the nominal span of the limbs.
///
/// [`combined_limbs_fit_modulus`] asks the span question, assuming every limb may be all ones. A
/// decomposition whose top limb is narrower than the others reaches only `2^value_bits`, not
/// `2^(count · limb_bits)`, so the span question would refuse representable widths: an `int253`
/// on bn254 splits into four 64-bit limbs spanning 256 bits, yet the value is one field element.
// FIELD-ASSUMPTION: L6-int-representation
// The result is one field element, so this is sound only while the recombined value fits one;
// this is the _representation_ half of the assumption, which field-derived limb widths do not
// discharge. The fit is checked, not assumed, so a narrow field refuses here instead of returning
// a residue. The refusal cannot fire: `passes::wide_witness_ints` holds any value wider than one
// cell carries injectively as limbs before a caller recombines it.
// FIELD-ASSUMPTION: L4-decompose
// The place values are minted as `two_pow`, so they are the intended powers only while they have
// not wrapped the modulus.
pub fn combine_limbs_of_value(
    b: &mut impl HLEmitter,
    limbs: &WitnessLimbs,
    value_bits: usize,
) -> ValueId {
    assert!(
        !limbs.limbs.is_empty(),
        "cannot recombine an empty decomposition"
    );
    if value_bits > widest_injective_int_bits(b.field()) {
        unsupported_on_this_field(
            format_args!(
                "recombining an int{value_bits} reaches 2^{value_bits} and so wraps modulo the field, leaving a residue no rangecheck on the result can tell from the value it should have been"
            ),
            b.field(),
        );
    }
    accumulate_limbs(b, limbs)
}

/// The place-value sum of a decomposition, with no bound applied.
///
/// Private because the bound is the whole question: every caller must have asked it, and which
/// form of the question is right depends on whether the top limb is full.
// FIELD-ASSUMPTION: L4-decompose
// The place values are minted as `two_pow`; see `combine_limbs_of_value`.
fn accumulate_limbs(b: &mut impl HLEmitter, limbs: &WitnessLimbs) -> ValueId {
    let mut fields = Vec::with_capacity(limbs.limbs.len());
    for &limb in &limbs.limbs {
        fields.push(b.cast_to_field(limb));
    }

    let mut acc = fields[0];
    for (i, &limb) in fields.iter().enumerate().skip(1) {
        let shift = b.field_const(b.field().two_pow(i * limbs.limb_bits));
        let shifted = b.umul(limb, shift);
        acc = b.uadd(acc, shifted);
    }
    acc
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mavros_int_semantics::IntBits;

    use crate::compiler::ssa::hlssa::{
        BinaryArithOpKind, Constant, HLSSA, MAX_SUPPORTED_INT_BITS, OpCode, builder::HLSSABuilder,
    };

    /// The LLVM lane's `Int <-> Field` packing has room the field does not carry.
    #[test]
    fn the_packing_has_room_the_field_does_not_carry() {
        let widest = widest_injective_int_bits(FieldConfig::bn254());

        assert!(crate::compiler::ssa::hlssa_to_llssa::INT_TO_FIELD_MAX_BITS > widest);
        assert!(!combined_limbs_fit_modulus(
            &field_modulus(FieldConfig::bn254()),
            crate::compiler::ssa::hlssa_to_llssa::INT_TO_FIELD_MAX_BITS,
            1
        ));
    }

    #[test]
    fn the_injective_width_is_where_a_single_limb_stops_fitting() {
        for modulus in [field_modulus(FieldConfig::bn254()), goldilocks()] {
            let widest = widest_injective_int_bits_for_modulus(&modulus);

            assert!(combined_limbs_fit_modulus(&modulus, widest, 1));
            assert!(!combined_limbs_fit_modulus(&modulus, widest + 1, 1));
        }
    }

    /// The answers for bn254 and goldilocks, written out.
    #[test]
    fn the_injective_widths_of_the_two_moduli() {
        assert_eq!(widest_injective_int_bits(FieldConfig::bn254()), 253);
        assert_eq!(widest_injective_int_bits(FieldConfig::goldilocks()), 63);
    }

    /// On bn254 the division's bound answers `2 · bits <= 253` at every width, so every bn254
    /// product, division and right shift takes the single cell exactly where a bit-length bound
    /// sends it.
    #[test]
    fn the_single_cell_bounds_are_a_bit_length_bound_on_bn254() {
        let bn254 = FieldConfig::bn254();
        let widest = widest_injective_int_bits(bn254);
        for bits in 1..=MAX_SUPPORTED_INT_BITS {
            let by_length = 2 * bits <= widest;
            let double_lane = bits == 2 * HOST_LIMB_BITS;
            assert_eq!(
                single_cell_product_fits(bn254, bits),
                by_length || double_lane,
                "int{bits} product"
            );
            assert_eq!(
                single_cell_signed_product_fits(bn254, bits),
                by_length,
                "int{bits} signed"
            );
            assert_eq!(
                single_cell_shift_fits(bn254, bits, false),
                by_length,
                "int{bits} right shift"
            );
        }
    }

    /// On goldilocks a `u32` division's `q·d + r` reaches `p - 1`, so the single cell takes every
    /// product, quotient and shift through 32 bits, and none at 33.
    #[test]
    fn goldilocks_divides_a_u32_in_one_cell_with_nothing_to_spare() {
        let p = goldilocks();
        let top = BigInt::one() << 32;
        assert_eq!(&top * (&top - BigInt::one()), &p - BigInt::one());

        let goldilocks_config = FieldConfig::goldilocks();
        for (bits, fits) in [(32, true), (33, false)] {
            assert_eq!(single_cell_product_fits(goldilocks_config, bits), fits);
            assert_eq!(
                single_cell_signed_product_fits(goldilocks_config, bits),
                fits
            );
            assert_eq!(single_cell_shift_fits(goldilocks_config, bits, false), fits);
            assert_eq!(single_cell_shift_fits(goldilocks_config, bits, true), fits);
        }
    }

    /// Goldilocks' modulus.
    fn goldilocks() -> BigInt {
        field_modulus(FieldConfig::goldilocks())
    }

    /// The widest multiple of `alignment` whose product with another limb is injective.
    ///
    /// A rule this module deliberately does **not** implement, for
    /// `a_coarser_granularity_would_buy_a_step_but_leave_a_ragged_limb` to compare against.
    fn coarser_limb_bits(modulus: &BigInt, alignment: usize) -> usize {
        let widest = widest_injective_int_bits_for_modulus(modulus);
        (alignment..=HOST_LIMB_BITS)
            .rev()
            .filter(|w| w % alignment == 0)
            .find(|&w| 2 * w <= widest)
            .expect("goldilocks admits at least a byte-wide limb")
    }

    #[test]
    fn bn254_is_capped_by_the_host_not_by_the_field() {
        // Every lowering's output depends on `h = 64` on bn254.
        let bn254 = FieldConfig::bn254();
        assert_eq!(witness_limb_bits(bn254), 64);
        assert_eq!(witness_half_limb_bits(bn254), 32);

        // The modulus alone would admit a limb past the host word.
        assert!(2 * (HOST_LIMB_BITS + 1) <= widest_injective_int_bits(bn254));
    }

    #[test]
    fn goldilocks_gets_a_16_bit_limb_because_a_32_bit_product_is_not_injective() {
        let goldilocks_config = FieldConfig::goldilocks();
        assert_eq!(witness_limb_bits(goldilocks_config), 16);
        assert_eq!(witness_half_limb_bits(goldilocks_config), 8);

        // The largest 32-bit product is below goldilocks, by exactly `2^32`, but has 64 bits
        // against an injective 63.
        let p = goldilocks();
        let max32 = (BigInt::one() << 32) - BigInt::one();
        assert!(&max32 * &max32 < p, "a bare 32x32 product fits goldilocks");
        assert_eq!(p.clone() - &max32 * &max32, BigInt::one() << 32);
        assert!(2 * 32 > widest_injective_int_bits(goldilocks_config));
    }

    #[test]
    fn a_coarser_granularity_would_buy_a_step_but_leave_a_ragged_limb() {
        // A byte-aligned rule would reach 24 on goldilocks, where powers of two stop at 16: four
        // limbs per u96 instead of six.
        let p = goldilocks();
        assert_eq!(coarser_limb_bits(&p, 8), 24);
        assert_eq!(limb_bits_for_modulus(&p), 16);

        // Not offered: 24 divides none of the widths lowered, so a u128 would widen to 144 bits,
        // past the narrow threshold.
        assert_ne!(32 % 24, 0);
        assert_ne!(128 % 24, 0);
        assert!(128usize.div_ceil(24) * 24 > narrow_int_bits(FieldConfig::bn254()));
    }

    /// The pow2 ceiling is a field question, which the two moduli answer differently. That bn254
    /// meets the cache-slot ceiling exactly is a coincidence.
    #[test]
    fn the_pow2_ceiling_is_the_widest_row_the_modulus_can_hold() {
        assert_eq!(max_pow2_table_size(FieldConfig::bn254()), 7);
        assert_eq!(max_pow2_table_size_for_modulus(&goldilocks()), 6);

        // Each answer and the size past it, so the boundary is pinned rather than a number.
        for modulus in [field_modulus(FieldConfig::bn254()), goldilocks()] {
            let size = max_pow2_table_size_for_modulus(&modulus);
            assert!(widest_pow2_row(size) < modulus);
            assert!(size == MAX_POW2_TABLE_SIZE || widest_pow2_row(size + 1) >= modulus);
        }
    }

    /// The threshold's two bounds **trade places** between the moduli, so neither can be dropped:
    /// on bn254 the host binds, on goldilocks the field.
    #[test]
    fn the_narrow_threshold_is_the_field_and_the_host_whichever_binds() {
        let bn254 = FieldConfig::bn254();
        assert_eq!(narrow_int_bits(bn254), 128);
        assert_eq!(narrow_int_bits_for_modulus(&goldilocks()), 63);

        // The property, at each answer and one past it.
        for modulus in [field_modulus(bn254), goldilocks()] {
            let n = narrow_int_bits_for_modulus(&modulus);
            assert!(
                combined_limbs_fit_modulus(&modulus, n, 1),
                "{n} does not fit"
            );
            assert!(
                n == NARROW_HOST_BITS || !combined_limbs_fit_modulus(&modulus, n + 1, 1),
                "{n} is not the widest that fits"
            );
        }

        assert!(
            narrow_int_bits(bn254) < MAX_SUPPORTED_INT_BITS,
            "the dispatch threshold and the type cap answer different questions, and should not answer them with the same number"
        );
        assert_eq!(narrow_int_bits(bn254), NARROW_HOST_BITS);
        assert!(narrow_int_bits_for_modulus(&goldilocks()) < NARROW_HOST_BITS);
    }

    /// Every width the narrow path admits has a `2^n` table, and so does every limb.
    ///
    /// `witness_bitwise::lower_shift` builds `2^n` for a witness `n` only from a table. The narrow
    /// half holds on goldilocks only because the threshold is derived: pinned at 128, it would ask
    /// for a size-7 table against a ceiling of 6.
    #[test]
    fn every_narrow_width_and_every_limb_has_a_powers_of_two_table() {
        for modulus in [field_modulus(FieldConfig::bn254()), goldilocks()] {
            let size = max_pow2_table_size_for_modulus(&modulus);
            let rows = 1usize << size;

            // The narrow half: 128 <= 128 on bn254, 63 <= 64 on goldilocks.
            let narrow = narrow_int_bits_for_modulus(&modulus);
            assert!(
                narrow <= rows,
                "narrow width {narrow} has no table at size {size}"
            );

            // A limb-wise shift shifts a limb by `0..h`, so the limb width needs a table.
            let h = limb_bits_for_modulus(&modulus);
            assert!(h <= rows, "limb width {h} has no table at size {size}");
        }
    }

    #[test]
    fn the_candidate_widths_are_the_powers_of_two_up_to_a_host_word() {
        let widths = candidate_widths();
        assert_eq!(*widths.first().unwrap(), HOST_LIMB_BITS);
        assert_eq!(*widths.last().unwrap(), NARROWEST_LIMB_BITS);
        assert!(widths.iter().all(|w| w.is_power_of_two()));
        assert!(widths.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn a_limb_product_is_injective_on_every_modulus() {
        // The defining property, checked across every modulus width a field could plausibly have.
        let widths = candidate_widths();
        for field_bits in 3..=256usize {
            // The smallest modulus a `field_bits`-wide field could have, the hardest case.
            let p = (BigInt::one() << (field_bits - 1)) + BigInt::one();
            let h = limb_bits_for_modulus(&p);
            assert!(
                widths.contains(&h),
                "{h} not admissible at {field_bits} bits"
            );

            // A `2h`-bit value, which every limb product is, is one distinct element ...
            assert!(
                combined_limbs_fit_modulus(&p, 2 * h, 1),
                "at {field_bits} bits"
            );
            // ... and nothing wider was passed over.
            for &wider in widths.iter().take_while(|&&w| w > h) {
                assert!(
                    !combined_limbs_fit_modulus(&p, 2 * wider, 1),
                    "{wider} was skipped at {field_bits} bits"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "has no witness limb")]
    fn a_field_below_four_has_no_limb() {
        limb_bits_for_modulus(&BigInt::from(3));
    }

    #[test]
    fn a_power_of_two_limb_divides_every_supported_integer_width() {
        // A power-of-two limb never leaves a ragged top limb on the power-of-two widths lowered;
        // see `a_coarser_granularity_would_buy_a_step_but_leave_a_ragged_limb`.
        for h in candidate_widths() {
            // Only widths a limb is actually cut out of; a limb wider than the value is one limb.
            for width in [8usize, 16, 32, 64, 128].into_iter().filter(|&w| w >= h) {
                assert_eq!(width % h, 0, "a {h}-bit limb straddles a u{width}");
            }
        }
    }

    #[test]
    fn a_two_limb_schoolbook_packing_asks_more_than_the_limb_rule_does() {
        // A field wide enough for a `2h`-bit product, all the limb rule asks, can be far too narrow
        // for `(2^h - 1)^2 * 2^(h+1)`.
        let h = 64usize;
        let just_fits_the_product = (BigInt::one() << 160) + BigInt::one();
        assert_eq!(
            limb_bits_for_modulus(&just_fits_the_product),
            h,
            "the limb rule holds at 160 bits"
        );
        assert!(!two_limb_packing_fits_modulus(&just_fits_the_product, h));

        // These bracket the packing's `2^(3h+1)`, pinning the `h + 1` exponent.
        assert!(!two_limb_packing_fits_modulus(
            &(BigInt::one() << (3 * h)),
            h
        ));
        assert!(two_limb_packing_fits_modulus(
            &(BigInt::one() << (3 * h + 1)),
            h
        ));

        // On bn254 the ~2^193 packing fits, but only for a width of exactly two limbs.
        let bn254 = FieldConfig::bn254();
        assert!(two_limb_product_packing_fits(bn254, 128));
        assert!(!two_limb_product_packing_fits(bn254, 64), "one limb");
        assert!(!two_limb_product_packing_fits(bn254, 192), "three limbs");
    }

    #[test]
    fn two_spreads_sum_inside_a_field_that_a_bit_count_bound_would_have_refused() {
        let bn254 = FieldConfig::bn254();

        // The half-limb and sub-64 bitwise widths fit bn254; the guard is for narrower fields.
        assert!(spread_sum_fits_field(witness_half_limb_bits(bn254), bn254));
        for bits in [2usize, 8, 16, 32, 63] {
            assert!(spread_sum_fits_field(bits, bn254), "{bits} bits");
        }

        // The bn254 boundary, far past what the `U(bits*2)` cast reaches: the cast is capped at
        // the narrow threshold, so no width past 64 could be spread there anyway.
        assert!(spread_sum_fits_field(127, bn254));
        assert!(!spread_sum_fits_field(128, bn254));
        assert!(2 * 65 > narrow_int_bits(bn254));

        // Against the modulus, goldilocks carries a 32-bit bitwise op; `2*bits < field_bits` would
        // stop at 31.
        let p = goldilocks();
        assert!(spread_sum_fits_modulus(32, &p));
        assert!(!spread_sum_fits_modulus(33, &p));
        assert!(2 * 32 >= 64, "the bit-count bound refuses this width");

        // A modulus can hold one 32-bit spread but not the sum of two.
        let holds_one_spread_only = BigInt::one() << 63;
        assert!(
            ((BigInt::one() << 64) - BigInt::one()) / 3 < holds_one_spread_only,
            "one 32-bit spread fits"
        );
        assert!(!spread_sum_fits_modulus(32, &holds_one_spread_only));
    }

    #[test]
    fn limbs_recombine_into_one_cell_only_while_their_place_values_fit() {
        let bn254 = field_modulus(FieldConfig::bn254());

        // The two-limb splits the lowerings make: a u64 from halves and a u128 from 64-bit limbs.
        assert!(combined_limbs_fit_modulus(&bn254, 32, 2), "u64 from halves");
        assert!(combined_limbs_fit_modulus(&bn254, 64, 2), "u128 from limbs");

        // The boundary is exact: a modulus of `2^(limb_count * limb_bits)` is the smallest that
        // holds every value.
        assert!(combined_limbs_fit_modulus(&(BigInt::one() << 64), 32, 2));
        assert!(!combined_limbs_fit_modulus(
            &((BigInt::one() << 64) - BigInt::one()),
            32,
            2
        ));

        // The limb width alone does not settle it: four goldilocks 16-bit limbs do not recombine.
        let p = goldilocks();
        assert_eq!(limb_bits_for_modulus(&p), 16);
        assert!(!combined_limbs_fit_modulus(&p, 16, 4));
        assert!(combined_limbs_fit_modulus(&p, 16, 3));
    }

    /// [`combine_limbs_of_value`] refuses a value wider than the field carries injectively, rather
    /// than returning a residue. No lowering reaches the refusal, so the test calls it directly.
    #[test]
    #[should_panic(expected = "recombining an int256 reaches 2^256")]
    fn a_recombination_too_wide_for_the_field_is_refused_rather_than_wrapped() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let limb = e.int_const(IntBits::one(64));
            combine_limbs_of_value(
                &mut e,
                &WitnessLimbs {
                    limb_bits: 64,
                    limbs: vec![limb; 4],
                },
                256,
            );
        });
    }

    #[test]
    fn a_three_limb_split_recombines_with_ascending_place_values() {
        // The caller reaches three limbs only from `int96` up, so the loop past the first place
        // value is pinned here.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let value = e.int_const(IntBits::from_u128(48, 0x0000_DEAD_BEEF));
                let limbs = split_into_limbs(&mut e, value, 16, 3);
                assert_eq!(limbs.limb_bits, 16);
                assert_eq!(limbs.limbs.len(), 3);
                combine_limbs_of_value(&mut e, &limbs, 48);
                e.terminate_return(vec![]);
            });
        }

        let function = ssa.get_unique_entrypoint();
        let ops: Vec<&OpCode> = function
            .get_block(function.get_entry_id())
            .get_instructions()
            .collect();

        // The split reads ascending, non-overlapping, equal-width windows.
        let windows: Vec<(usize, usize)> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::BitRange { offset, width, .. } => Some((*offset, *width)),
                _ => None,
            })
            .collect();
        assert_eq!(windows, vec![(0, 16), (16, 16), (32, 16)]);

        // And the recombination scales limb `i` by `2^(16i)` -- one multiply fewer than there are
        // limbs, because limb 0 carries no place value at all.
        let place_values: Vec<BigInt> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::UMul,
                    rhs,
                    ..
                } => Some(*rhs),
                _ => None,
            })
            .map(|rhs| match ssa.get_const(rhs).as_deref() {
                Some(Constant::Field(f)) => field_to_bigint_via_bytes(f),
                other => panic!("a place value must be a field constant, got {other:?}"),
            })
            .collect();
        assert_eq!(
            place_values,
            vec![BigInt::one() << 16, BigInt::one() << 32],
            "place values must be 2^h and 2^2h, not 2^h twice"
        );
    }

    /// The canonical integer behind a field element, as a `BigInt`.
    fn field_to_bigint_via_bytes(f: &crate::compiler::Field) -> BigInt {
        use num_bigint::Sign;
        let bytes: Vec<u8> = f
            .into_bigint()
            .0
            .iter()
            .flat_map(|limb| limb.to_le_bytes())
            .collect();
        BigInt::from_bytes_le(Sign::Plus, &bytes)
    }
}
