use crate::compiler::ssa::ValueId;
use num_traits::{One, Zero};

use mavros_int_semantics::IntBits;
use num_bigint::BigUint;

#[derive(Clone)]
pub(crate) struct FPolynomial<'a> {
    /// The degree of the `F` polynomial
    pub(crate) degree: usize,

    /// The terms `F` sums, each `±lhs·rhs` or `±lhs` of factors.
    pub(super) terms: Vec<Term<'a>>,

    /// What a polynomial identity solves for;
    /// or [`None`] where every term of the polynomial is given.
    pub(super) answer: Option<Answer<'a>>,

    /// The blocks `0..t` in order.
    pub(crate) blocks: Vec<Block>,

    /// What the identity is checked with.
    params: FPolynomialParams,
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
    let carry_check_bits = max_carry.bits() as usize;

    if is_checked {
        let group_size = injective_integer_width.checked_sub(carry_check_bits)? / low_limbs_width;
        (group_size > 0).then_some(FPolynomialParams {
            carry_check_bits,
            carry_offset_bits: None,
            group_size,
            t: k,
            pure_bits: low_limbs_width + carry_check_bits.max(low_limbs_width + 1),
        })
    } else {
        let group_size =
            injective_integer_width.saturating_sub(carry_check_bits + 1) / low_limbs_width;
        (t == 0 || group_size > 0).then_some(FPolynomialParams {
            carry_check_bits: carry_check_bits + 1,
            carry_offset_bits: Some(carry_check_bits),
            group_size,
            t,
            pure_bits: low_limbs_width + carry_check_bits.max(low_limbs_width) + 1,
        })
    }
}

/// One term of an identity.
#[derive(Clone, Copy)]
pub(in super::super) struct Term<'a> {
    pub(in super::super) polarity: Polarity,
    pub(in super::super) lhs: &'a Factor,
    pub(in super::super) rhs: Option<&'a Factor>,
    pub(in super::super) gated: bool,
}

/// Whether a term of an identity is added or subtracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Polarity {
    Positive,
    Negative,
}

impl Polarity {
    pub(in super::super) fn flipped(self) -> Self {
        match self {
            Polarity::Positive => Polarity::Negative,
            Polarity::Negative => Polarity::Positive,
        }
    }
}

/// What a checked polynomial solves for, either a given answer or on to be solved.
#[derive(Clone, Copy)]
pub(in super::super) enum Answer<'a> {
    /// The answer to be solved.
    Unknown(&'a [usize]),

    /// A given factor, held to the columns by the polynomial.
    Given(&'a Factor),
}

impl<'a> FPolynomial<'a> {
    /// The columns the carry chain writes a witness from: a carry out of each committed one, and
    /// an answer limb from each of a witnessed answer's.
    pub(crate) fn carry_chain_len(&self) -> usize {
        let answer_limbs = match self.answer {
            Some(Answer::Unknown(widths)) => widths.len(),
            Some(Answer::Given(_)) | None => 0,
        };
        self.num_committed_carries().max(answer_limbs)
    }

    pub(crate) fn num_committed_carries(&self) -> usize {
        if self.is_checked() {
            self.degree / 2
        } else {
            self.degree
        }
    }

    /// Whether the identity is checked against a known answer.
    fn is_checked(&self) -> bool {
        self.answer.is_some()
    }

    pub(crate) fn carry_offset_bits(&self) -> Option<usize> {
        self.params.carry_offset_bits
    }

    pub(crate) fn hint_bits(&self) -> usize {
        self.params.pure_bits
    }
}

pub(crate) fn build_f_polynomial<'a>(
    widths: &[usize],
    injective: usize,
    terms: Vec<Term<'a>>,
    answer: Option<Answer<'a>>,
) -> Option<FPolynomial<'a>> {
    let products = terms.iter().filter(|term| term.rhs.is_some()).count();
    let params = f_polynomial_params(
        widths,
        injective,
        products,
        terms.len() - products,
        answer.is_some(),
    )?;
    // A product reaches `2k − 1` columns, and a single factor `k`.
    let k = widths.len();
    let degree = if products > 0 { 2 * k - 2 } else { k - 1 };
    if params.t > degree + 1 {
        return None;
    }
    let mut identity = FPolynomial {
        degree,
        terms,
        answer,
        blocks: Vec::new(),
        params,
    };
    let committed_carries = identity.num_committed_carries();
    let certified = params.t;
    let group_size = params.group_size.max(1);
    identity.blocks = (0..certified)
        .step_by(group_size)
        .map(|first| {
            let last = (first + group_size).min(certified) - 1;
            Block {
                first,
                last,
                carry_bits: (last < committed_carries).then_some(params.carry_check_bits),
            }
        })
        .collect();
    Some(identity)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    pub(crate) first: usize,
    pub(crate) last: usize,
    pub(crate) carry_bits: Option<usize>,
}

/// One operand of the schoolbook, limb by limb.
#[allow(non_snake_case)]
pub(crate) struct Factor {
    /// Each limb at its width.
    pub(crate) limbs: Vec<Limb>,

    /// Whether the operand is witnessed, which decides how the hints of its [`Limb::Value`]s are
    /// read. A witnessed operand can still have known limbs, which are [`Limb::Constant`]s.
    pub(crate) is_witnessed: bool,

    /// The largest value each limb can take: the largest its width holds, or a constant limb's value.
    pub(crate) bounds: Vec<BigUint>,

    /// `L = 2^(w−1)` for a signed factor where `w` is the top limb width.
    pub(crate) L: Option<BigUint>,
}

impl Factor {
    /// `limbs` at `widths` as a factor
    #[allow(non_snake_case)]
    pub(crate) fn new(
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

    /// `L` where `limb` is the top limb and the factor is signed.
    pub(super) fn top_offset(&self, limb: usize) -> Option<&BigUint> {
        self.L.as_ref().filter(|_| limb + 1 == self.limbs.len())
    }

    /// Whether coefficient `limb` is zero by construction: its limb's bound is zero and is not offset.
    pub(super) fn vanishes(&self, limb: usize) -> bool {
        self.bounds[limb].is_zero() && self.top_offset(limb).is_none()
    }

    /// The largest value the whole operand can take, from its limbs' bounds at `limb_bits` apart.
    pub(crate) fn largest(&self, limb_bits: usize) -> BigUint {
        self.bounds
            .iter()
            .enumerate()
            .map(|(index, bound)| bound << (index * limb_bits))
            .sum()
    }

    /// The smallest value the whole operand can take: its known limbs, with every other at zero.
    pub(crate) fn smallest(&self, limb_bits: usize) -> BigUint {
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
pub(crate) enum Limb {
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
