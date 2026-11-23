use std::{fmt::Debug, iter::once};

use crate::compiler::ssa::ValueId;
use num_traits::{One, Zero};

use mavros_int_semantics::IntBits;
use num_bigint::BigUint;

mod lower;

/// An identity as the limb config judges it.
pub(in super::super) trait IdentityShape: Debug + Sync {
    /// Whether the identity holds over operands cut as `layout`.
    fn does_identity_hold(&self, layout: LimbLayout, injective: usize) -> bool;
}

/// Every identity is judged by laying out its own terms, over full-range factors.
impl<T: IdentityTemplate> IdentityShape for T {
    fn does_identity_hold(&self, layout: LimbLayout, injective: usize) -> bool {
        let full = Factor::full(layout);
        let (terms, target) = self.largest_terms(&full);
        chain_layout(&terms, target, injective).is_some()
    }
}

/// An identity the lowering builds.
pub(super) trait IdentityTemplate: Copy + Debug + Sync {
    /// What it is built from: the factors known before it is lowered, and the [`Target`] it is
    /// held to.
    type Factors<'a>;

    /// Its terms over `factors`, and the target they are held to.
    fn terms<'a>(self, factors: Self::Factors<'a>) -> (Vec<Term<'a>>, Option<Target<'a>>);

    /// Its terms at their largest, over `full`, a full-range factor.
    fn largest_terms<'a>(self, full: &'a Factor) -> (Vec<Term<'a>>, Option<Target<'a>>);

    /// The identity this lays out over `factors`; or [`None`] where the field cannot hold it.
    fn instantiate<'a>(
        self,
        injective: usize,
        factors: Self::Factors<'a>,
    ) -> Option<PolynomialIdentity<'a>> {
        let (terms, target) = self.terms(factors);
        let layout = chain_layout(&terms, target, injective)?;
        Some(PolynomialIdentity {
            terms,
            target,
            layout,
        })
    }
}

#[derive(Clone)]
/// The polynomial representation of a wide integer identity.
/// The equivalent of `F(X)` in [this scheme](https://xn--2-umb.com/26/dependent-r1cs/#integer-embedding)
pub(super) struct PolynomialIdentity<'a> {
    /// The terms the identity sums, each of the form `±A(X)·B(X)` or `A(X)` of polynomial factors.
    /// Each of these factors represents one wide integer.
    pub(super) terms: Vec<Term<'a>>,

    /// What a polynomial identity solves for: e.g C(x) when the identity of the `A(x)B(x) - C(x) = 0`;
    /// or [`None`] where every term of the polynomial is given, and the identity can be of any shape.
    pub(super) target: Option<Target<'a>>,

    /// How the carry chain is represented in R1CS.
    layout: ChainLayout,
}

impl PolynomialIdentity<'_> {
    /// The columns the carry chain writes a witness from: a carry out of each committed one, and
    /// a target limb from each of a witnessed target's.
    pub(super) fn carry_chain_len(&self) -> usize {
        let target_limbs = match self.target {
            Some(Target::Witnessed(answer)) => answer.num_limbs(),
            Some(Target::Given(_)) | None => 0,
        };
        self.num_committed_carries().max(target_limbs)
    }

    /// The degree of the `F` polynomial.
    pub(super) fn degree(&self) -> usize {
        self.layout.degree
    }

    /// The blocks `0..t` in order.
    pub(super) fn blocks(&self) -> &[Block] {
        &self.layout.blocks
    }

    /// The number of carries that are actually witnessed. `k - 1` for a checked identity,
    /// or [`PolynomialIdentity::degree`] otherwise.
    pub(super) fn num_committed_carries(&self) -> usize {
        self.layout.num_committed_carries
    }

    pub(super) fn carry_offset_bits(&self) -> Option<usize> {
        self.layout.carry_offset_bits
    }

    pub(super) fn hint_bits(&self) -> usize {
        self.layout.hint_bits
    }
}

/// One term of an identity.
#[derive(Clone, Copy)]
pub(super) struct Term<'a> {
    /// The sign of the term in the identity: positive or negative
    pub(super) polarity: Polarity,
    /// Compulsory polynomial factor in the term.
    pub(super) lhs: &'a Factor,
    /// The second factor of a quadratic term: `B(X)` in `A(X)·B(X)`; or [`None`] for a linear term.
    pub(super) rhs: Option<&'a Factor>,
    /// The term should be zeroed out when the [`PolynomialIdentity`] it belongs to is run under a guard
    /// and the guard is false.
    ///
    /// Some terms may be supplied as zero by design when the guard is false,
    /// so no need to add extra operations to zero them out.
    pub(super) gated: bool,
}

impl Term<'_> {
    /// Its factors: `A`, and `B` where it is a product.
    fn factors(&self) -> impl Iterator<Item = &Factor> {
        once(self.lhs).chain(self.rhs)
    }

    /// The degree of its polynomial.
    fn degree(&self) -> usize {
        self.factors()
            .map(|factor| factor.layout.num_limbs() - 1)
            .sum()
    }

    /// The largest value it can take, at full range.
    fn largest(&self) -> BigUint {
        self.factors()
            .map(|factor| max_limb_value(factor.layout.bits))
            .product()
    }

    /// The most it can carry out of any column.
    fn max_carry(&self) -> BigUint {
        match self.rhs {
            Some(rhs) => max_product_carry(self.lhs.layout, rhs.layout),
            // A linear term's limbs below a column, read at their places, are below the column's
            // place, so it carries less than one out of it.
            None => BigUint::one(),
        }
    }
}

/// Whether a term of an identity is added or subtracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Polarity {
    Positive,
    Negative,
}

impl Polarity {
    pub(super) fn flipped(self) -> Self {
        match self {
            Polarity::Positive => Polarity::Negative,
            Polarity::Negative => Polarity::Positive,
        }
    }
}

/// What a checked polynomial is held to: a value it witnesses, or a given factor.
#[derive(Clone, Copy)]
pub(super) enum Target<'a> {
    /// A value the carry chain witnesses.
    Witnessed(LimbLayout),

    /// A given factor, held to the columns by the polynomial.
    Given(&'a Factor),
}

impl Target<'_> {
    /// How the value the identity is held to is cut into limbs.
    fn layout(&self) -> LimbLayout {
        match *self {
            Target::Witnessed(layout) => layout,
            Target::Given(factor) => factor.layout,
        }
    }
}

/// How an integer of `bits` is cut into limbs of `limb_bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct LimbLayout {
    /// The width of the whole integer.
    pub(super) bits: usize,

    /// The width of every limb but the top one.
    pub(super) low_limb_bits: usize,
}

impl LimbLayout {
    pub(in super::super) fn new(bits: usize, low_limb_bits: usize) -> Self {
        LimbLayout {
            bits,
            low_limb_bits,
        }
    }

    /// The number of limbs.
    fn num_limbs(self) -> usize {
        self.bits.div_ceil(self.low_limb_bits)
    }

    /// The place of `limb`: the power of two it is scaled by.
    fn place(self, limb: usize) -> usize {
        limb * self.low_limb_bits
    }

    /// The width of `limb`: `limb_bits`, or what is left of `bits` for the top one.
    fn width(self, limb: usize) -> usize {
        (self.bits - self.place(limb)).min(self.low_limb_bits)
    }
}

/// One operand of the schoolbook, limb by limb.
pub(super) struct Factor {
    /// How the operand is cut into limbs.
    pub(super) layout: LimbLayout,

    /// Each limb, at its place in `layout`.
    pub(super) limbs: Vec<Limb>,

    /// Whether the operand is witnessed, which decides how the hints of its [`Limb::Value`]s are
    /// read. A witnessed operand can still have known limbs, which are [`Limb::Constant`]s.
    pub(super) is_witnessed: bool,

    /// The largest value each limb can take: the largest its width holds, or a constant limb's value.
    pub(super) bounds: Vec<BigUint>,

    /// `L = 2^(w−1)` for a signed factor where `w` is the top limb width.
    pub(super) signed_offset: Option<BigUint>,
}

impl Factor {
    /// `limbs`, cut as `layout`, as a factor.
    pub(super) fn new(
        layout: LimbLayout,
        limbs: Vec<Limb>,
        is_witnessed: bool,
        signed_offset: Option<BigUint>,
    ) -> Self {
        if limbs.len() != layout.num_limbs() {
            ice_unreachable!(
                "a factor of {} limbs cut as {layout:?}, which has {}",
                limbs.len(),
                layout.num_limbs()
            );
        }
        let bounds = limbs
            .iter()
            .enumerate()
            .map(|(index, limb)| match limb {
                Limb::Constant(pattern) => BigUint::from(pattern),
                Limb::Value(_) => max_limb_value(layout.width(index)),
            })
            .collect();
        Factor {
            layout,
            limbs,
            is_witnessed,
            bounds,
            signed_offset,
        }
    }

    /// A full-range witnessed factor cut as `layout`.
    pub(super) fn full(layout: LimbLayout) -> Self {
        let limbs = (0..layout.num_limbs())
            .map(|_| Limb::Value(ValueId(0)))
            .collect();
        Factor::new(layout, limbs, true, None)
    }

    /// It with each limb bounded by `bounds`, narrower than its width.
    pub(super) fn with_bounds(self, bounds: Vec<BigUint>) -> Self {
        Factor { bounds, ..self }
    }

    /// `L` where `limb` is the top limb and the factor is signed.
    pub(super) fn top_offset(&self, limb: usize) -> Option<&BigUint> {
        self.signed_offset
            .as_ref()
            .filter(|_| limb + 1 == self.limbs.len())
    }

    /// Whether coefficient `limb` is zero by construction: its limb's bound is zero and is not offset.
    pub(super) fn vanishes(&self, limb: usize) -> bool {
        self.bounds[limb].is_zero() && self.top_offset(limb).is_none()
    }

    /// The largest value the whole operand can take, from its limbs' bounds at their places.
    pub(super) fn largest(&self) -> BigUint {
        self.bounds
            .iter()
            .enumerate()
            .map(|(index, bound)| bound << self.layout.place(index))
            .sum()
    }

    /// The smallest value the whole operand can take: its known limbs, with every other at zero.
    pub(super) fn smallest(&self) -> BigUint {
        self.limbs
            .iter()
            .enumerate()
            .filter_map(|(index, limb)| match limb {
                Limb::Constant(pattern) => Some(BigUint::from(pattern) << self.layout.place(index)),
                Limb::Value(_) => None,
            })
            .sum()
    }
}

/// One limb of a [`Factor`].
pub(super) enum Limb {
    /// A value, witnessed or pure as the operand is.
    Value(ValueId),

    /// A limb of a constant operand, interned only where the plan reads it.
    Constant(IntBits),
}

/// How an identity is cut into blocks.
#[derive(Clone)]
struct ChainLayout {
    /// The degree of the identity polynomial.
    degree: usize,

    /// The carries the chain commits.
    num_committed_carries: usize,

    /// The blocks `0..t`.
    blocks: Vec<Block>,

    /// `Some(c)` when a carry can be negative; carries are then witnessed raised by `2^c`.
    carry_offset_bits: Option<usize>,

    /// The integer width the pure side computes at.
    hint_bits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One block of columns
/// Carries between columns in the block cancel; they are witnessed but not range-checked.
pub(super) struct Block {
    /// Index of first column in the block.
    /// `a` in the notation of [this scheme](https://xn--2-umb.com/26/dependent-r1cs/#integer-embedding)
    pub(super) first: usize,
    /// Index of last column in the block.
    /// `b` in the notation of [this scheme](https://xn--2-umb.com/26/dependent-r1cs/#integer-embedding)
    pub(super) last: usize,
    /// Number of bits allowed to be carried out of the block. [`None`] where the block ends at
    /// the top column, which carries nothing out.
    pub(super) carry_bits: Option<usize>,
}

/// The layout of the identity of `terms` held to `target`; or [`None`] where it is inadmissible.
fn chain_layout(
    terms: &[Term<'_>],
    target: Option<Target<'_>>,
    injective: usize,
) -> Option<ChainLayout> {
    let params = check_params(terms, target, injective)?;
    let degree = degree(terms);
    if params.t > degree + 1 {
        return None;
    }
    let num_committed_carries = match target {
        Some(target) => target.layout().num_limbs() - 1,
        None => degree,
    };
    let certified = params.t;
    let group_size = params.group_size.max(1);
    let blocks = (0..certified)
        .step_by(group_size)
        .map(|first| {
            let last = (first + group_size).min(certified) - 1;
            Block {
                first,
                last,
                carry_bits: (last < num_committed_carries).then_some(params.carry_check_bits),
            }
        })
        .collect();
    Some(ChainLayout {
        degree,
        num_committed_carries,
        blocks,
        carry_offset_bits: params.carry_offset_bits,
        hint_bits: params.hint_bits,
    })
}

/// What an identity is checked with, from [`check_params`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CheckParams {
    /// The width each carry at a block boundary is range-checked at.
    carry_check_bits: usize,

    /// `Some(c)` when a carry can be negative, which is then witnessed raised by `2^c`.
    carry_offset_bits: Option<usize>,

    /// `g`: the columns per block.
    group_size: usize,

    /// `t`: the number of low columns the identity soundly constrains, `F(β) ≡ 0 (mod β^t)`.
    /// For a checked identity `t` = `k`.
    t: usize,

    /// The integer width the pure side computes at.
    hint_bits: usize,
}

/// What the identity of `terms` held to `target` is checked with.
fn check_params(
    terms: &[Term<'_>],
    target: Option<Target<'_>>,
    injective_integer_width: usize,
) -> Option<CheckParams> {
    let limb_bits = terms
        .first()
        .expect("an identity has a term")
        .lhs
        .layout
        .low_limb_bits;
    let mut layouts = terms
        .iter()
        .flat_map(Term::factors)
        .map(|factor| factor.layout)
        .chain(target.map(|target| target.layout()));
    if !layouts.all(|layout| layout.low_limb_bits == limb_bits && layout.num_limbs() >= 2) {
        ice_unreachable!(
            "polynomial evaluation expects every factor in multiple limbs at one limb width"
        );
    }

    let max_polynomial_value: BigUint = terms.iter().map(Term::largest).sum();

    let t = (max_polynomial_value.bits() as usize)
        .saturating_sub(injective_integer_width)
        .div_ceil(limb_bits);

    let max_carry: BigUint = terms.iter().map(Term::max_carry).sum();
    let carry_bits = max_carry.bits() as usize;
    let has_negative_terms = terms.iter().any(|term| term.polarity == Polarity::Negative);
    let (carry_check_bits, carry_offset_bits, sign_bits) = if has_negative_terms {
        (carry_bits + 1, Some(carry_bits), 1)
    } else {
        (carry_bits, None, 0)
    };

    if let Some(target) = target {
        let group_size = injective_integer_width.checked_sub(carry_check_bits)? / limb_bits;
        (group_size > 0).then_some(CheckParams {
            carry_check_bits,
            carry_offset_bits,
            group_size,
            t: target.layout().num_limbs(),
            hint_bits: limb_bits + carry_bits.max(limb_bits + 1) + sign_bits,
        })
    } else {
        let group_size = injective_integer_width.saturating_sub(carry_check_bits) / limb_bits;
        (t == 0 || group_size > 0).then_some(CheckParams {
            carry_check_bits,
            carry_offset_bits,
            group_size,
            t,
            hint_bits: limb_bits + carry_bits.max(limb_bits) + sign_bits,
        })
    }
}

/// The max that partial products of factors cut as `lhs` and `rhs` can carry out of any column.
fn max_product_carry(lhs: LimbLayout, rhs: LimbLayout) -> BigUint {
    if lhs == rhs {
        let k = lhs.num_limbs();
        let low_limbs_max_value = max_limb_value(lhs.low_limb_bits);
        let top_limb_max_value = max_limb_value(lhs.width(k - 1));
        return (&low_limbs_max_value * (k - 1))
            .max(&low_limbs_max_value * (k - 2) + top_limb_max_value * 2u8);
    }
    let limb_sum = |layout: LimbLayout| -> BigUint {
        (0..layout.num_limbs())
            .map(|limb| max_limb_value(layout.width(limb)))
            .sum()
    };
    limb_sum(lhs).min(limb_sum(rhs))
}

/// The largest value a limb of `width` holds.
fn max_limb_value(width: usize) -> BigUint {
    (BigUint::one() << width) - 1u8
}

/// The degree of `F`: that of its highest term.
fn degree(terms: &[Term<'_>]) -> usize {
    terms
        .iter()
        .map(Term::degree)
        .max()
        .expect("an identity has a term")
}

#[cfg(test)]
mod tests {
    use super::{ChainLayout, Factor, IdentityTemplate, LimbLayout, chain_layout};
    use crate::compiler::passes::{
        shared::limbs::{widest_injective_int_bits, witness_limb_bits},
        wide_witness_ints::{
            rewriter::{
                division::EuclideanDivision, product::UnsignedProduct, sign::SignedProduct,
            },
            test_utils::bn254,
        },
    };
    use mavros_artifacts::FieldConfig;

    /// Every block of every identity, at every width, groups no more columns
    /// than one element can sum together with the carry.
    #[test]
    fn every_block_of_an_identity_fits_an_element_with_its_carry() {
        for field in [bn254(), FieldConfig::goldilocks()] {
            let id = field.id();
            let injective = widest_injective_int_bits(field);
            let h = witness_limb_bits(field);
            for bits in [h + 1, 2 * h, 3 * h - 1, 320, 1000, 16384] {
                let limbs = LimbLayout::new(bits, h);
                let what = format!("{id:?} int{bits}");
                check_shape(UnsignedProduct, injective, limbs, &what);
                check_shape(SignedProduct, injective, limbs, &what);
                check_shape(EuclideanDivision, injective, limbs, &what);
            }
        }
    }

    /// Check the block invariants of `shape` laid out over full-range factors cut as `limbs`.
    fn check_shape<T: IdentityTemplate>(shape: T, injective: usize, limbs: LimbLayout, what: &str) {
        let full = Factor::full(limbs);
        let (terms, target) = shape.largest_terms(&full);
        check_blocks(
            chain_layout(&terms, target, injective),
            target.is_some(),
            injective,
            limbs,
            &format!("{what} {shape:?}"),
        );
    }

    /// Check the block invariants.
    fn check_blocks(
        layout: Option<ChainLayout>,
        is_checked: bool,
        injective: usize,
        limbs: LimbLayout,
        what: &str,
    ) {
        let layout = layout.unwrap_or_else(|| panic!("{what}: the field holds the identity"));
        let limb_bits = limbs.low_limb_bits;
        let mut next = 0;
        for block in &layout.blocks {
            assert_eq!(
                block.first, next,
                "{what}: blocks are contiguous from column 0"
            );
            assert!(block.last >= block.first, "{what}: no block is empty");
            next = block.last + 1;
            assert_eq!(
                block.carry_bits.is_some(),
                block.last < layout.num_committed_carries,
                "{what}: block {block:?} checks its carry exactly where one is committed"
            );
            if let Some(carry_bits) = block.carry_bits {
                let columns = block.last - block.first + 1;
                assert!(
                    columns * limb_bits + carry_bits <= injective,
                    "{what}: block {block:?} of {columns} columns with a {carry_bits}-bit carry \
                     exceeds the {injective}-bit injective width"
                );
            }
        }
        if is_checked {
            assert_eq!(
                next,
                limbs.num_limbs(),
                "{what}: a checked identity certifies every column"
            );
        }
        assert!(
            next <= layout.degree + 1,
            "{what}: no more columns certified than F has"
        );
    }
}
