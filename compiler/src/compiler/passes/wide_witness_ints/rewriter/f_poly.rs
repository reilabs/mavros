use std::fmt;

use crate::compiler::ssa::ValueId;
use num_traits::{One, Zero};

use mavros_int_semantics::IntBits;
use num_bigint::BigUint;

#[derive(Clone)]
pub(super) struct FPolynomial<'a> {
    /// The terms `F` sums, each `±lhs·rhs` or `±lhs` of factors.
    pub(super) terms: Vec<Term<'a>>,

    /// What a polynomial identity solves for;
    /// or [`None`] where every term of the polynomial is given.
    pub(super) answer: Option<Answer<'a>>,

    layout: Layout,
}

/// The shape of an identity. Used for finding sound parameters.
pub(in super::super) trait IdentityShape: fmt::Debug + Sync {
    /// The number of `±lhs·rhs` terms.
    fn num_products(&self) -> usize;

    /// The number of `±lhs` terms.
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
        layout(self, widths, injective).is_some()
    }
}

/// The degree of `F` at `k` limbs.
fn degree(products: usize, k: usize) -> usize {
    if products > 0 { 2 * k - 2 } else { k - 1 }
}

/// What an identity is checked with, from [`f_polynomial_params`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FPolynomialParams {
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
    pure_bits: usize,
}

/// What an identity over integers at limb `widths` is checked with.
fn f_polynomial_params(
    widths: &[usize],
    injective_integer_width: usize,
    num_quadratic_terms: usize,
    num_linear_terms: usize,
    is_checked: bool,
    has_negative_terms: bool,
) -> Option<FPolynomialParams> {
    let k = widths.len();
    if k < 2 {
        ice_unreachable!("polynomial evluation expects multiple limbs");
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
        (group_size > 0).then_some(FPolynomialParams {
            carry_check_bits,
            carry_offset_bits,
            group_size,
            t: k,
            pure_bits: low_limbs_width + carry_bits.max(low_limbs_width + 1) + sign_bits,
        })
    } else {
        let group_size = injective_integer_width.saturating_sub(carry_check_bits) / low_limbs_width;
        (t == 0 || group_size > 0).then_some(FPolynomialParams {
            carry_check_bits,
            carry_offset_bits,
            group_size,
            t,
            pure_bits: low_limbs_width + carry_bits.max(low_limbs_width) + sign_bits,
        })
    }
}

/// One term of an identity.
#[derive(Clone, Copy)]
pub(super) struct Term<'a> {
    pub(super) polarity: Polarity,
    pub(super) lhs: &'a Factor,
    pub(super) rhs: Option<&'a Factor>,
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

/// What a checked polynomial solves for, either a given answer or on to be solved.
#[derive(Clone, Copy)]
pub(super) enum Answer<'a> {
    /// The answer to be solved.
    Unknown(&'a [usize]),

    /// A given factor, held to the columns by the polynomial.
    Given(&'a Factor),
}

impl FPolynomial<'_> {
    /// The columns the carry chain writes a witness from: a carry out of each committed one, and
    /// an answer limb from each of a witnessed answer's.
    pub(super) fn carry_chain_len(&self) -> usize {
        let answer_limbs = match self.answer {
            Some(Answer::Unknown(widths)) => widths.len(),
            Some(Answer::Given(_)) | None => 0,
        };
        self.num_committed_carries().max(answer_limbs)
    }

    /// The degree of the `F` polynomial.
    pub(super) fn degree(&self) -> usize {
        self.layout.degree
    }

    /// The blocks `0..t` in order.
    pub(super) fn blocks(&self) -> &[Block] {
        &self.layout.blocks
    }

    pub(super) fn num_committed_carries(&self) -> usize {
        self.layout.num_committed_carries
    }

    pub(super) fn carry_offset_bits(&self) -> Option<usize> {
        self.layout.carry_offset_bits
    }

    pub(super) fn hint_bits(&self) -> usize {
        self.layout.pure_bits
    }
}

/// An identity the lowering builds, which lays out its own terms, so they are of its shape.
pub(super) trait Identity<'a>: IdentityShape + Copy {
    /// What it is built from.
    type Factors;

    /// Its terms over `factors`, and the answer they are held to.
    fn terms(self, factors: Self::Factors) -> (Vec<Term<'a>>, Option<Answer<'a>>);
}

/// The identity `shape` lays out over `factors` at limb `widths`; or [`None`] exactly where
/// `shape` does not [admit](IdentityShape::admits) `widths`.
pub(super) fn build_f_polynomial<'a, S: Identity<'a>>(
    shape: S,
    widths: &[usize],
    injective: usize,
    factors: S::Factors,
) -> Option<FPolynomial<'a>> {
    let layout = layout(&shape, widths, injective)?;
    let (terms, answer) = shape.terms(factors);
    if !shape.has_negative_terms() && terms.iter().any(|t| t.polarity == Polarity::Negative) {
        ice!("{shape:?} subtracts a term, but its carries are planned as non-negative");
    }
    Some(FPolynomial {
        terms,
        answer,
        layout,
    })
}

/// How an identity is cut into blocks.
#[derive(Clone)]
struct Layout {
    /// The degree of the `F` polynomial.
    degree: usize,

    /// The carries the chain commits.
    num_committed_carries: usize,

    /// The blocks `0..t`.
    blocks: Vec<Block>,

    /// `Some(c)` when a carry can be negative; carries are then witnessed raised by `2^c`.
    carry_offset_bits: Option<usize>,

    /// The integer width the pure side computes at.
    pure_bits: usize,
}

/// The layout of an identity of `shape` at limb `widths`; or [`None`] where it is inadmissable.
fn layout<S: IdentityShape + ?Sized>(
    shape: &S,
    widths: &[usize],
    injective: usize,
) -> Option<Layout> {
    let params = f_polynomial_params(
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
    Some(Layout {
        degree,
        num_committed_carries,
        blocks,
        carry_offset_bits: params.carry_offset_bits,
        pure_bits: params.pure_bits,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Block {
    pub(super) first: usize,
    pub(super) last: usize,
    pub(super) carry_bits: Option<usize>,
}

/// One operand of the schoolbook, limb by limb.
#[allow(non_snake_case)]
pub(super) struct Factor {
    /// Each limb at its width.
    pub(super) limbs: Vec<Limb>,

    /// Whether the operand is witnessed, which decides how the hints of its [`Limb::Value`]s are
    /// read. A witnessed operand can still have known limbs, which are [`Limb::Constant`]s.
    pub(super) is_witnessed: bool,

    /// The largest value each limb can take: the largest its width holds, or a constant limb's value.
    pub(super) bounds: Vec<BigUint>,

    /// `L = 2^(w−1)` for a signed factor where `w` is the top limb width.
    pub(super) L: Option<BigUint>,
}

impl Factor {
    /// `limbs` at `widths` as a factor
    #[allow(non_snake_case)]
    pub(super) fn new(
        limbs: Vec<Limb>,
        widths: &[usize],
        is_witnessed: bool,
        L: Option<BigUint>,
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
            L,
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
        self.L.as_ref().filter(|_| limb + 1 == self.limbs.len())
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

#[cfg(test)]
mod tests {
    use super::{IdentityShape, Layout, layout};
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
                        layout(shape, &widths, injective),
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
        layout: Option<Layout>,
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
