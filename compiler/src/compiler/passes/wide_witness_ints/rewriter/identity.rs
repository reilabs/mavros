use std::fmt;

use crate::compiler::ssa::ValueId;
use num_traits::{One, Zero};

use mavros_int_semantics::IntBits;
use num_bigint::BigUint;

/// The shape of an identity. Used for finding sound parameters.
pub(in super::super) trait IdentityShape: fmt::Debug + Sync {
    /// The number of `±A(X).B(X)` terms.
    fn num_products(&self) -> usize;

    /// The number of `±D(X)` terms.
    fn num_linear_terms(&self) -> usize;

    /// Whether it is checked against an answer.
    fn is_checked(&self) -> bool {
        false
    }

    /// Whether any of its terms is subtracted, so that a carry can be negative.
    fn has_negative_terms(&self) -> bool {
        false
    }

    /// Whether an identity of this shape holds at limb `widths`.
    fn admits(&self, widths: &[usize], injective: usize) -> bool {
        chain_layout(self, widths, injective).is_some()
    }
}

/// An identity the lowering builds, which lays out its own terms, so they are of its shape.
pub(super) trait IdentityTemplate<'a>: IdentityShape + Copy {
    /// What it is built from.
    type Factors;

    /// Its terms over `factors`, and the target they are held to.
    fn terms(self, factors: Self::Factors) -> (Vec<Term<'a>>, Option<Target<'a>>);

    /// The identity this lays out over `factors` at limb `widths`; or [`None`] exactly where
    /// it does not [admit](IdentityShape::admits) `widths`.
    fn instantiate(
        self,
        widths: &[usize],
        injective: usize,
        factors: Self::Factors,
    ) -> Option<PolynomialIdentity<'a>> {
        let layout = chain_layout(&self, widths, injective)?;
        let (terms, target) = self.terms(factors);
        if !self.has_negative_terms() && terms.iter().any(|t| t.polarity == Polarity::Negative) {
            ice!("{self:?} subtracts a term, but its carries are planned as non-negative");
        }
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
            Some(Target::Witnessed(widths)) => widths.len(),
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
    Witnessed(&'a [usize]),

    /// A given factor, held to the columns by the polynomial.
    Given(&'a Factor),
}

/// One operand of the schoolbook, limb by limb.
pub(super) struct Factor {
    /// Each limb at its width.
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
    /// `limbs` at `widths` as a factor
    pub(super) fn new(
        limbs: Vec<Limb>,
        widths: &[usize],
        is_witnessed: bool,
        signed_offset: Option<BigUint>,
    ) -> Self {
        let bounds = limbs
            .iter()
            .zip(full_bounds(widths))
            .map(|(limb, full)| match limb {
                Limb::Constant(pattern) => BigUint::from(pattern),
                Limb::Value(_) => full,
            })
            .collect();
        Factor {
            limbs,
            is_witnessed,
            bounds,
            signed_offset,
        }
    }

    /// A full-range witnessed factor at limb `widths`.
    #[cfg(test)]
    pub(super) fn full(widths: &[usize]) -> Self {
        let limbs = widths.iter().map(|_| Limb::Value(ValueId(0))).collect();
        Factor::new(limbs, widths, true, None)
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

    /// The largest value the whole operand can take, from its limbs' bounds at `limb_bits` apart.
    pub(super) fn largest(&self, limb_bits: usize) -> BigUint {
        self.bounds
            .iter()
            .enumerate()
            .map(|(index, bound)| bound << (index * limb_bits))
            .sum()
    }

    /// The smallest value the whole operand can take: its known limbs, with every other at zero.
    pub(super) fn smallest(&self, limb_bits: usize) -> BigUint {
        self.limbs
            .iter()
            .enumerate()
            .filter_map(|(index, limb)| match limb {
                Limb::Constant(pattern) => Some(BigUint::from(pattern) << (index * limb_bits)),
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

/// The bound of each limb at `widths` when nothing narrower is known: the largest value its width
/// holds.
fn full_bounds(widths: &[usize]) -> Vec<BigUint> {
    widths
        .iter()
        .map(|width| (BigUint::one() << *width) - 1u8)
        .collect()
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

/// The layout of an identity of `shape` at limb `widths`; or [`None`] where it is inadmissible.
fn chain_layout<S: IdentityShape + ?Sized>(
    shape: &S,
    widths: &[usize],
    injective: usize,
) -> Option<ChainLayout> {
    let params = check_params(
        widths,
        injective,
        shape.num_products(),
        shape.num_linear_terms(),
        shape.is_checked(),
        shape.has_negative_terms(),
    )?;
    let degree = degree(shape.num_products(), widths.len());
    if params.t > degree + 1 {
        return None;
    }
    let num_committed_carries = if shape.is_checked() {
        widths.len() - 1
    } else {
        degree
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

/// What an identity over integers at limb `widths` is checked with.
fn check_params(
    widths: &[usize],
    injective_integer_width: usize,
    num_quadratic_terms: usize,
    num_linear_terms: usize,
    is_checked: bool,
    has_negative_terms: bool,
) -> Option<CheckParams> {
    let k = widths.len();
    if k < 2 {
        ice_unreachable!("polynomial evaluation expects multiple limbs");
    }
    let get_max_limb_value = |width: usize| (BigUint::one() << width) - 1u8;

    let (low_limbs_width, low_limbs_max_value, top_limb_max_value) = (
        widths[0], // All limbs have same width except the top limb.
        get_max_limb_value(widths[0]),
        get_max_limb_value(widths[k - 1]),
    );
    let max_operand_value = get_max_limb_value(widths.iter().sum());
    let max_polynomial_value = &max_operand_value * &max_operand_value * num_quadratic_terms
        + &max_operand_value * num_linear_terms;

    let t = (max_polynomial_value.bits() as usize)
        .saturating_sub(injective_integer_width)
        .div_ceil(low_limbs_width);

    let max_partial_product_carry = (&low_limbs_max_value * (k - 1))
        .max(&low_limbs_max_value * (k - 2) + &top_limb_max_value * 2u8);
    let max_carry = max_partial_product_carry * num_quadratic_terms + num_linear_terms;
    let carry_bits = max_carry.bits() as usize;
    let (carry_check_bits, carry_offset_bits, sign_bits) = if has_negative_terms {
        (carry_bits + 1, Some(carry_bits), 1)
    } else {
        (carry_bits, None, 0)
    };

    if is_checked {
        let group_size = injective_integer_width.checked_sub(carry_check_bits)? / low_limbs_width;
        (group_size > 0).then_some(CheckParams {
            carry_check_bits,
            carry_offset_bits,
            group_size,
            t: k,
            hint_bits: low_limbs_width + carry_bits.max(low_limbs_width + 1) + sign_bits,
        })
    } else {
        let group_size = injective_integer_width.saturating_sub(carry_check_bits) / low_limbs_width;
        (t == 0 || group_size > 0).then_some(CheckParams {
            carry_check_bits,
            carry_offset_bits,
            group_size,
            t,
            hint_bits: low_limbs_width + carry_bits.max(low_limbs_width) + sign_bits,
        })
    }
}

/// The degree of `F` at `k` limbs.
fn degree(num_quadratic_terms: usize, k: usize) -> usize {
    if num_quadratic_terms > 0 {
        2 * k - 2
    } else {
        k - 1
    }
}

#[cfg(test)]
mod tests {
    use super::{ChainLayout, IdentityShape, chain_layout};
    use crate::compiler::passes::{
        shared::limbs::{widest_injective_int_bits, witness_limb_bits},
        wide_witness_ints::{
            rewriter::{
                division::EuclideanDivision, product::UnsignedProduct, sign::SignedProduct,
            },
            test_utils::bn254,
            types::limb_widths,
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
                let widths = limb_widths(bits, h);
                let shapes: [&dyn IdentityShape; 3] =
                    [&UnsignedProduct, &SignedProduct, &EuclideanDivision];
                for shape in shapes {
                    check_blocks(
                        chain_layout(shape, &widths, injective),
                        shape.is_checked(),
                        injective,
                        &widths,
                        &format!("{id:?} int{bits} {shape:?}"),
                    );
                }
            }
        }
    }

    /// Check the block invariants.
    fn check_blocks(
        layout: Option<ChainLayout>,
        is_checked: bool,
        injective: usize,
        widths: &[usize],
        what: &str,
    ) {
        let layout = layout.unwrap_or_else(|| panic!("{what}: the field holds the identity"));
        let limb_bits = widths[0];
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
                widths.len(),
                "{what}: a checked identity certifies every column"
            );
        }
        assert!(
            next <= layout.degree + 1,
            "{what}: no more columns certified than F has"
        );
    }
}
