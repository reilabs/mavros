//! The witness limb: how wide a piece of an integer can be for a given field.
//!
//! This governs witness layout and the widths the spread-based bitwise lowering works at. It is
//! also the width the schoolbook multiplier and its carry chain will be built at when P5 lands —
//! see `docs/field-agnosticism.md`, Layer 6.

use num_bigint::BigInt;
use num_traits::One;

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::int_bits::HOST_LIMB_BITS;

use crate::compiler::{
    analysis::value_range_analysis::field_modulus,
    passes::shared::unsupported::unsupported_on_this_field,
    ssa::{
        ValueId,
        hlssa::{CastTarget, builder::HLEmitter},
    },
};

// THE WITNESS LIMB WIDTH
// ================================================================================================

/// The width to fall back to when a field is too small to carry any limb at all.
///
/// Also the narrowest admissible width, so [`limb_bits_for_modulus`] is total: every field gets an
/// answer, and the width guards downstream are the ones that refuse.
const NARROWEST_LIMB_BITS: usize = 1;

/// What a witness limb has to be able to fit: the parameters that determine its width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimbBudget {
    /// How many `h`-bit partial products must fit in one field element at once.
    ///
    /// `1` asks only that a bare `a·b` not wrap: a limb is exactly what multiplies in-field, and
    /// nothing more. Each doubling buys one more product per reduction and costs a narrower limb,
    /// so the trade is **reductions against limb count**: a bigger budget means more limbs per
    /// value and more partial products overall, but longer runs between witnessed, range-checked
    /// carry extractions as part of the reduction.
    ///
    /// At `1` a schoolbook `acc += a·b` does not fit whenever this is the binding constraint, so
    /// such a field's accumulator has to reduce after every product rather than per column.
    ///
    /// This counts **products**, not place-value headroom. A lowering that scales a column by `2^h`
    /// before adding it, spreads its operands, or recombines its limbs needs a strictly stronger
    /// condition than any setting here states. Those conditions are the separate predicates listed
    /// on [`witness_limb_bits`].
    pub accumulated_products: u32,
}

impl LimbBudget {
    /// The budget the compiler actually runs on.
    ///
    /// `accumulated_products: 1` buys the **widest** limb the field allows, which is the trade the
    /// corpus suggests: its lookup traffic is 8-, 16- and 32-bit, not long chains of multiply and
    /// accumulate.
    pub const DEFAULT: LimbBudget = LimbBudget {
        accumulated_products: 1,
    };
}

/// The admissible limb widths, widest first: the powers of two up to [`HOST_LIMB_BITS`].
///
/// Never empty — [`NARROWEST_LIMB_BITS`] is always the last entry.
///
/// The ceiling is [`HOST_LIMB_BITS`] and is deliberately **not** a budget knob: a limb is carried
/// in a host word and an `IntBits` limb, so it is pinned by the representation, not chosen. Making
/// it settable would let a caller ask for a limb that has nowhere to live.
fn candidate_widths() -> Vec<usize> {
    (NARROWEST_LIMB_BITS..=HOST_LIMB_BITS)
        .rev()
        .filter(|w| w.is_power_of_two())
        .collect()
}

/// The witness limb width `h` for `field`.
///
/// `h` is the width a value is chopped into before its pieces are packed into field elements. It
/// governs the splits in `witness_bitwise` and `witness_integer_arith`.
///
/// See [`limb_bits_for_modulus`] for the rule and [`LimbBudget`] for the tuning tools. Each of
/// these is a separate predicate because each asks for strictly more than one bare `a·b`:
///
/// - that a lowering which scales a limb column by a place value still fits
///   ([`two_limb_product_packing_fits`]);
/// - that the sum of two spreads still fits ([`spread_sum_fits_field`]);
/// - that the limbs recombine into one cell ([`combined_limbs_fit_field`]).
pub fn witness_limb_bits(field: FieldConfig) -> usize {
    limb_bits_for_modulus(&field_modulus(field), LimbBudget::DEFAULT)
}

/// The widest admissible limb whose partial products still fit `modulus`, under `budget`.
///
/// The predicate is stated against the **modulus itself**, not its bit length, necessary to take
/// full advantage of potential limb sizes. Goldilocks, for example, is `2^64 - 2^32 + 1`, so
/// `(2^32 - 1)^2` clears it by exactly `2^32` and a bit-count bound of the form `2h < field_bits`
/// would reject a 32-bit limb that fits.
pub fn limb_bits_for_modulus(modulus: &BigInt, budget: LimbBudget) -> usize {
    candidate_widths()
        .into_iter()
        .find(|&limb_bits| {
            let max_limb = (BigInt::one() << limb_bits) - BigInt::one();
            let budgeted = (&max_limb * &max_limb) * BigInt::from(budget.accumulated_products);
            budgeted < *modulus
        })
        // A field too small for even the narrowest limb is not one this compiler can lower for at
        // all; returning the narrowest lets the width guards downstream be the ones that refuse.
        .unwrap_or(NARROWEST_LIMB_BITS)
}

/// Half a witness limb: the width the spread-based bitwise lowering operates at.
///
/// `Spread` interleaves zero bits, so the spread of a `k`-bit value occupies `2k` bits. Taking
/// `k = h/2` is what makes a spread occupy exactly one limb, which is why 64-bit bitwise splits
/// into 32-bit pieces on bn254 rather than spreading the word whole.
///
/// # Panics
///
/// If the limb has no exact half. The halves have to cover the limb between them: `derive_low_limb`
/// range-checks the low half to exactly this many bits, so an odd `h` would leave the top bit of
/// the low half unconstrained. [`LimbBudget`] admits only powers of two precisely so this holds,
/// and the assertion pins that the two have not drifted apart.
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
/// Two things have to hold:
///
/// - `bits` has to (for now) decompose into exactly two limbs on this field, because the lowering
///   reads the split as a pair. On a narrower field the same width yields more limbs and there is
///   no two-limb product to pack.
/// - The packed column has to fit. [`witness_limb_bits`] certifies that **one** partial product
///   fits, which is all [`LimbBudget::accumulated_products`] can express. A schoolbook
///   `lo·lo + (lo·hi + hi·lo)·2^h` asks for more, because the place value scales a two-product
///   column. See [`two_limb_packing_fits_modulus`] for that half.
pub fn two_limb_product_packing_fits(field: FieldConfig, bits: usize) -> bool {
    let limb_bits = witness_limb_bits(field);
    bits.div_ceil(limb_bits) == 2 && two_limb_packing_fits_modulus(&field_modulus(field), limb_bits)
}

/// Whether `lo·lo + (lo·hi + hi·lo)·2^h` stays below `modulus` for limbs of `limb_bits` bits.
///
/// With limbs bounded by `M = 2^h − 1` the packed value reaches `M²·(2^{h+1} + 1)`, roughly
/// `2^{3h+1}`. On bn254 that is ~2^193 against a ~2^254 modulus, but it does not follow from the
/// limb width: a field wide enough for `M²` — which is all the budget asks — can still be far too
/// narrow for `M²·2^{h+1}`.
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
/// `Spread` interleaves zero bits, so a `bits`-wide spread occupies the even positions below
/// `2*bits` and is at most `(4^bits − 1)/3`; two of them sum to at most twice that.
///
/// Stated against the modulus rather than its bit length for the same reason
/// [`limb_bits_for_modulus`] is: the sum is barely over two thirds of `2^(2*bits)`, so a bound of
/// the form `2*bits < field_bits` gives away a whole width. It is the difference between accepting
/// and refusing a bare `u32` bitwise op on goldilocks, which `docs/field-agnosticism.md` records
/// as fitting.
pub fn spread_sum_fits_modulus(bits: usize, modulus: &BigInt) -> bool {
    let max_spread = ((BigInt::one() << (2 * bits)) - BigInt::one()) / 3;
    (max_spread << 1) < *modulus
}

/// Whether `limb_count` limbs of `limb_bits` bits each recombine into one element of `field`.
///
/// This is [`combine_limbs`]' own precondition, and the representation half of the same question
/// the two predicates above ask about operations: the recombined value reaches
/// `2^(limb_count · limb_bits) − 1`, and once that can exceed the modulus the sum is a residue and
/// nothing downstream can tell it from the honest value.
pub fn combined_limbs_fit_field(field: FieldConfig, limb_bits: usize, limb_count: usize) -> bool {
    combined_limbs_fit_modulus(&field_modulus(field), limb_bits, limb_count)
}

/// Whether `limb_count` limbs of `limb_bits` bits each recombine below `modulus`.
///
/// The limbs are bounded by their own width, so the recombination is below `2^(limb_count ·
/// limb_bits)`; that bound is tight, since every limb may be all ones.
pub fn combined_limbs_fit_modulus(modulus: &BigInt, limb_bits: usize, limb_count: usize) -> bool {
    (BigInt::one() << (limb_bits * limb_count)) <= *modulus
}

// LIMB DECOMPOSITIONS
// ================================================================================================

/// A value decomposed into equal-width limbs, least significant first.
///
/// Equal-width is a contract, but does **not** forbid slack in the top limb. A limb carrying fewer
/// meaningful bits than its width, the rest provably zero, is an ordinary `Int(limb_bits)` like
/// every other one.
///
/// The limbs are usually `Int(limb_bits)` values, but [`combine_limbs`] also accepts already-field
/// limbs — `witness_bitwise`'s 128-bit path recombines two field-valued halves that way — so a
/// consumer should not assume the integer typing without checking.
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

/// Recombine `limbs` into a single field value, `limb[0] + limb[1] * 2^h + ...`.
///
/// Refuses on a field the recombination does not fit; see [`combined_limbs_fit_field`].
// FIELD-ASSUMPTION: L6-int-representation
// The result is one field element, so this is sound only while the recombined value fits one — it
// is the _representation_ half of the assumption that survives limb widths becoming field-derived.
// The widths are now `h` and the fit is now checked rather than assumed, so a narrow field refuses
// here instead of returning a residue; what is still missing is the multi-cell representation that
// would let it _succeed_, which is Phase 4's work rather than a wider constant.
// FIELD-ASSUMPTION: L4-decompose
// The place values are minted as `two_pow`, so they are only the powers they are meant to be while
// they have not wrapped the modulus.
pub fn combine_limbs(b: &mut impl HLEmitter, limbs: &WitnessLimbs) -> ValueId {
    assert!(
        !limbs.limbs.is_empty(),
        "cannot recombine an empty decomposition"
    );
    if !combined_limbs_fit_field(b.field(), limbs.limb_bits, limbs.limbs.len()) {
        unsupported_on_this_field(
            format_args!(
                "recombining {} limbs of {} bits reaches 2^{} and so wraps modulo the field, leaving a residue no rangecheck on the result can tell from the value it should have been",
                limbs.limbs.len(),
                limbs.limb_bits,
                limbs.limbs.len() * limbs.limb_bits
            ),
            b.field(),
        );
    }
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

/// Derive the low limb of `value` by subtracting an already-placed high limb, as a field value.
// FIELD-ASSUMPTION: L4-decompose
// The place value is minted as `two_pow`; see `combine_limbs`.
pub fn derive_low_limb(
    b: &mut impl HLEmitter,
    value: ValueId,
    hi_field: ValueId,
    limb_bits: usize,
) -> ValueId {
    let value_field = b.cast_to_field(value);
    let shift = b.field_const(b.field().two_pow(limb_bits));
    let shifted_hi = b.umul(hi_field, shift);
    let lo_field = b.usub(value_field, shifted_hi);
    b.cast_to(CastTarget::Int(limb_bits), lo_field)
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use crate::compiler::ssa::hlssa::{
        BinaryArithOpKind, Constant, HLSSA, MAX_SUPPORTED_UNSIGNED_BITS, OpCode,
        builder::HLSSABuilder,
    };

    /// `2^64 - 2^32 + 1`. Not a field this compiler can be configured for yet, which is exactly why
    /// it is written out here: it is the worked example in `docs/field-agnosticism.md`, and the only
    /// modulus on which the tuning knob makes a visible difference.
    fn goldilocks() -> BigInt {
        (BigInt::one() << 64) - (BigInt::one() << 32) + BigInt::one()
    }

    /// The default budget with `accumulated_products` overridden.
    fn products(count: u32) -> LimbBudget {
        LimbBudget {
            accumulated_products: count,
        }
    }

    /// The widest width that is a multiple of `alignment` and whose budgeted product fits `modulus`.
    ///
    /// The rule this module deliberately does **not** implement, kept as a local so
    /// `a_coarser_granularity_would_buy_a_step_but_leave_a_ragged_limb` can state what is being
    /// given up without offering it as a setting.
    fn coarser_limb_bits(modulus: &BigInt, alignment: usize, count: u32) -> usize {
        (alignment..=HOST_LIMB_BITS)
            .rev()
            .filter(|w| w % alignment == 0)
            .find(|&w| {
                let max_limb = (BigInt::one() << w) - BigInt::one();
                &max_limb * &max_limb * BigInt::from(count) < *modulus
            })
            .expect("goldilocks admits at least a byte-wide limb")
    }

    #[test]
    fn bn254_is_capped_by_the_host_not_by_the_field() {
        // The number the whole of Phase 2 is held to: `h = 64` on bn254 is what keeps every
        // existing lowering byte-identical.
        assert_eq!(witness_limb_bits(FieldConfig::bn254()), 64);
        assert_eq!(witness_half_limb_bits(FieldConfig::bn254()), 32);

        // And it is the cap that binds, not the modulus: bn254 has room for a far wider limb, so
        // every knob setting short of an absurd accumulation budget gives the same answer.
        let p = field_modulus(FieldConfig::bn254());
        for count in [1u32, 2, 16, 1 << 20] {
            assert_eq!(
                limb_bits_for_modulus(&p, products(count)),
                HOST_LIMB_BITS,
                "bn254 at {count} products"
            );
        }
    }

    #[test]
    fn goldilocks_gets_a_32_bit_limb_because_a_bare_product_fits() {
        // The fact a bit-count bound cannot see, and the reason the predicate is stated against the
        // modulus: the largest 32-bit product clears goldilocks by exactly `2^32`.
        let p = goldilocks();
        let max32 = (BigInt::one() << 32) - BigInt::one();
        assert!(&max32 * &max32 < p, "a bare 32x32 product fits goldilocks");
        assert_eq!(p.clone() - &max32 * &max32, BigInt::one() << 32);

        // So asking only that `a*b` not wrap answers 32, which is what the default asks ...
        assert_eq!(limb_bits_for_modulus(&p, products(1)), 32);
        assert_eq!(limb_bits_for_modulus(&p, LimbBudget::DEFAULT), 32);
        // ... and asking for room to accumulate a second product drops it to 16, because there is
        // no power of two in between. This one step is the whole content of the knob on goldilocks,
        // and the reason `accumulated_products` is the knob worth having.
        assert_eq!(limb_bits_for_modulus(&p, products(2)), 16);
    }

    #[test]
    fn a_coarser_granularity_would_buy_a_step_but_leave_a_ragged_limb() {
        // The temptation, stated so the decision is on the record rather than implied by an absent
        // setting: a byte-aligned rule reaches 24 on goldilocks where powers of two stop at 16,
        // which is four limbs per u96 instead of six.
        let p = goldilocks();
        assert_eq!(coarser_limb_bits(&p, 8, 2), 24);
        assert_eq!(limb_bits_for_modulus(&p, products(2)), 16);

        // And the reason it is not offered: 24 does not divide the widths this compiler lowers, so
        // every split would first have to widen its operand to a multiple of 24 -- which for a u128
        // is 144 bits, past the integer type cap. `lookup_sizing` charges the slack on top.
        assert_ne!(32 % 24, 0);
        assert_ne!(128 % 24, 0);
        assert!(128usize.div_ceil(24) * 24 > MAX_SUPPORTED_UNSIGNED_BITS);

        // And nothing is given up at the budget the compiler actually runs: 24 only wins once a
        // second product has to fit, and at `DEFAULT` the two rules agree on 32, which is a power
        // of two _and_ byte-aligned. Dropping the granularity knob costs goldilocks nothing.
        assert_eq!(coarser_limb_bits(&p, 8, 1), 32);
        assert_eq!(limb_bits_for_modulus(&p, LimbBudget::DEFAULT), 32);
    }

    #[test]
    fn the_default_budget_is_the_one_the_compiler_runs_on() {
        // `witness_limb_bits` is a thin wrapper; pin that it really is the default budget, so a
        // knob edit cannot move production while leaving the knob tests green.
        let p = field_modulus(FieldConfig::bn254());
        assert_eq!(
            witness_limb_bits(FieldConfig::bn254()),
            limb_bits_for_modulus(&p, LimbBudget::DEFAULT)
        );
        assert_eq!(LimbBudget::DEFAULT.accumulated_products, 1);
    }

    #[test]
    fn the_candidate_widths_are_the_powers_of_two_up_to_a_host_word() {
        let widths = candidate_widths();
        // The ceiling is not a knob: no budget can ask for a limb wider than a host word.
        assert_eq!(*widths.first().unwrap(), HOST_LIMB_BITS);
        // And the list is never empty, which is what makes `limb_bits_for_modulus` total.
        assert_eq!(*widths.last().unwrap(), NARROWEST_LIMB_BITS);
        assert!(widths.iter().all(|w| w.is_power_of_two()));
        assert!(widths.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn a_limb_always_multiplies_inside_the_field_within_its_budget() {
        // The defining property, checked rather than trusted across every modulus width a field
        // could plausibly have.
        let widths = candidate_widths();
        for b in [LimbBudget::DEFAULT, products(2), products(7)] {
            for field_bits in 2..=256usize {
                // A modulus just above the low end of its bit range, which is the hardest case: it
                // is the smallest prime-sized value a `field_bits`-wide field could carry.
                let p = (BigInt::one() << (field_bits - 1)) + BigInt::one();
                let h = limb_bits_for_modulus(&p, b);
                assert!(
                    widths.contains(&h),
                    "{h} not admissible at {field_bits} bits"
                );
                assert!(h <= HOST_LIMB_BITS, "at {field_bits} bits");

                let fits = |w: usize| {
                    let m = (BigInt::one() << w) - BigInt::one();
                    &m * &m * BigInt::from(b.accumulated_products) < p
                };
                // Either the limb is genuinely admissible, or the field is too small for even the
                // narrowest one and we fell back to it.
                assert!(fits(h) || h == NARROWEST_LIMB_BITS, "at {field_bits} bits");
                // And nothing wider was passed over.
                for &wider in widths.iter().take_while(|&&w| w > h) {
                    assert!(!fits(wider), "{wider} was skipped at {field_bits} bits");
                }
            }
        }
    }

    #[test]
    fn a_power_of_two_limb_divides_every_supported_integer_width() {
        // Why the granularity is fixed at powers of two: the widths this compiler actually lowers
        // are powers of two, so a power-of-two limb never leaves a ragged top limb, whatever the
        // field. See `a_coarser_granularity_would_buy_a_step_but_leave_a_ragged_limb` for the
        // counter-example, and `analysis::lookup_sizing` for what the residue would cost.
        for h in candidate_widths() {
            // Only widths a limb is actually cut out of; a limb wider than the value is one limb.
            for width in [8usize, 16, 32, 64, 128].into_iter().filter(|&w| w >= h) {
                assert_eq!(width % h, 0, "a {h}-bit limb straddles a u{width}");
            }
        }
    }

    #[test]
    fn a_two_limb_schoolbook_packing_asks_more_than_the_budget_does() {
        // The gap the predicate exists for, driven through the function itself rather than through
        // a re-derivation of its formula: a field wide enough for `(2^h - 1)^2` -- which is all the
        // budget asks -- can still be far too narrow for `(2^h - 1)^2 * 2^(h+1)`.
        let h = 64usize;
        let just_fits_the_product = (BigInt::one() << 160) + BigInt::one();
        assert_eq!(
            limb_bits_for_modulus(&just_fits_the_product, LimbBudget::DEFAULT),
            h,
            "the budget's own condition holds at 160 bits"
        );
        assert!(!two_limb_packing_fits_modulus(&just_fits_the_product, h));

        // The packing needs about `2^(3h+1)`, and these two powers bracket where it lands. They are
        // what makes the `h + 1` exponent load-bearing: drop it to `h` and the packing fits a
        // modulus a whole limb width smaller.
        assert!(!two_limb_packing_fits_modulus(
            &(BigInt::one() << (3 * h)),
            h
        ));
        assert!(two_limb_packing_fits_modulus(
            &(BigInt::one() << (3 * h + 1)),
            h
        ));

        // And what `witness_limb_bits` does buy on the field that exists: bn254's limb is
        // host-capped at 64 and the ~2^193 packing has room to spare -- but only for a width that
        // really is two limbs, which is the other half of the question.
        let bn254 = FieldConfig::bn254();
        assert!(two_limb_product_packing_fits(bn254, 128));
        assert!(!two_limb_product_packing_fits(bn254, 64), "one limb");
        assert!(!two_limb_product_packing_fits(bn254, 192), "three limbs");
    }

    #[test]
    fn two_spreads_sum_inside_a_field_that_a_bit_count_bound_would_have_refused() {
        let bn254 = FieldConfig::bn254();

        // The width the half-limb decomposition delivers, and the widths the sub-64 bitwise
        // dispatch reaches directly: all clear bn254 with room to spare, which is why this guard is
        // a statement about a narrower field rather than about today's lowering.
        assert!(spread_sum_fits_field(witness_half_limb_bits(bn254), bn254));
        for bits in [2usize, 8, 16, 32, 63] {
            assert!(spread_sum_fits_field(bits, bn254), "{bits} bits");
        }

        // The boundary on bn254, well past the `U(bits*2)` cast's own reach: the cast is capped at
        // `MAX_SUPPORTED_UNSIGNED_BITS`, so no width past 64 could have been spread there anyway.
        assert!(spread_sum_fits_field(127, bn254));
        assert!(!spread_sum_fits_field(128, bn254));
        assert!(2 * 65 > MAX_SUPPORTED_UNSIGNED_BITS);

        // Why the bound is stated against the modulus. A spread pair is barely over two thirds of
        // `2^(2*bits)`, so goldilocks carries a full 32-bit bitwise op -- which is what
        // `docs/field-agnosticism.md` records -- while `2*bits < field_bits` would have stopped at
        // 31 and refused it.
        let p = goldilocks();
        assert!(spread_sum_fits_modulus(32, &p));
        assert!(!spread_sum_fits_modulus(33, &p));
        assert!(2 * 32 >= 64, "the bit-count bound refuses this width");

        // And why it is the sum of *two* spreads rather than one: a modulus can hold a single
        // 32-bit spread and not the pair, which is the only thing that separates this bound from
        // the bound on one operand.
        let holds_one_spread_only = BigInt::one() << 63;
        assert!(
            ((BigInt::one() << 64) - BigInt::one()) / 3 < holds_one_spread_only,
            "one 32-bit spread fits"
        );
        assert!(!spread_sum_fits_modulus(32, &holds_one_spread_only));
    }

    #[test]
    fn limbs_recombine_into_one_cell_only_while_their_place_values_fit() {
        let bn254 = FieldConfig::bn254();

        // Every recombination in the tree, at the widths it actually runs at.
        assert!(combined_limbs_fit_field(bn254, 32, 2), "u64 from halves");
        assert!(combined_limbs_fit_field(bn254, 64, 2), "u128 from limbs");

        // The boundary is exact rather than approximate: `limb_count` limbs of `limb_bits` reach
        // `2^(limb_count * limb_bits) - 1`, so a modulus of exactly that power is the last one that
        // still holds every representative.
        assert!(combined_limbs_fit_modulus(&(BigInt::one() << 64), 32, 2));
        assert!(!combined_limbs_fit_modulus(
            &((BigInt::one() << 64) - BigInt::one()),
            32,
            2
        ));

        // The window the guard exists for, and the reason the limb width alone does not close it:
        // goldilocks admits a 32-bit limb, and two of them do not recombine.
        let p = goldilocks();
        assert_eq!(limb_bits_for_modulus(&p, LimbBudget::DEFAULT), 32);
        assert!(!combined_limbs_fit_modulus(&p, 32, 2));
        assert!(combined_limbs_fit_modulus(&p, 32, 1));
    }

    /// The predicate above is wired into [`combine_limbs`], not merely available beside it.
    ///
    /// Unlike the other refusals in the tree this one is reachable on bn254 — not from any lowering
    /// the compiler has, but from a decomposition wide enough to ask for it, which is what the
    /// multi-cell work will eventually build.
    #[test]
    #[should_panic(expected = "recombining 4 limbs of 64 bits reaches 2^256")]
    fn a_recombination_too_wide_for_the_field_is_refused_rather_than_wrapped() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let limb = e.int_const(64, 1);
            combine_limbs(
                &mut e,
                &WitnessLimbs {
                    limb_bits: 64,
                    limbs: vec![limb; 4],
                },
            );
        });
    }

    #[test]
    fn a_three_limb_split_recombines_with_ascending_place_values() {
        // Every caller in the tree splits into exactly two, so `combine_limbs`' loop past the first
        // place value -- and the `i * limb_bits` exponent it mints -- has no other coverage.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let value = e.int_const(48, 0x0000_DEAD_BEEF);
                let limbs = split_into_limbs(&mut e, value, 16, 3);
                assert_eq!(limbs.limb_bits, 16);
                assert_eq!(limbs.limbs.len(), 3);
                combine_limbs(&mut e, &limbs);
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
