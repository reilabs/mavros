use crate::compiler::ssa::ValueId;
use num_traits::{One, Zero};

use mavros_int_semantics::IntBits;
use num_bigint::BigUint;
/// One operand of the schoolbook, limb by limb.
pub(crate) struct Factor {
    /// Each limb at its width.
    pub(crate) limbs: Vec<Limb>,

    /// Whether the operand is witnessed, which decides how the hints of its [`Limb::Value`]s are
    /// read. A witnessed operand can still have known limbs, which are [`Limb::Constant`]s.
    pub(crate) witnessed: bool,

    /// The largest value each limb can take: the largest its width holds, or a constant limb's value.
    pub(crate) bounds: Vec<BigUint>,
}

impl Factor {
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

/// One step in accumulating a column of the product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Add the partial product of limb `lhs` of the left operand and limb `rhs` of the right.
    Product { lhs: usize, rhs: usize },

    /// Add limb `index` of the addend, which lands in the column of the same index.
    Addend(usize),

    /// Add the next carry out of the column below, in the order that column made them.
    Carry,

    /// Bound the accumulator.
    ///
    /// Below the top column it splits into its low limb, which stays in the column, and a witnessed
    /// carry of `carry_bits` into the next; the top column may carry nothing out, so there it is
    /// range-checked at the top limb's width instead.
    Reduce { carry_bits: Option<usize> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Block {
    pub(super) first: usize,
    pub(super) last: usize,
    pub(super) carry_bits: Option<usize>,
}

/// How a product's columns are accumulated and where they are reduced, decided on the operands'
/// bounds alone.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ProductPlan {
    /// The steps of each column of the answer, little-endian. Empty where the product is
    /// evaluated.
    ///
    /// A column whose bound passes its limb's width ends in a reduction, and the accumulator left
    /// afterwards is that limb.
    pub(super) columns: Vec<Vec<Step>>,

    /// The overflow check.
    ///
    /// Each left limb with the right limbs whose partial products with it land at or past the top
    /// of the answer, grouped so that one group's sum times the limb stays inside an element. Empty
    /// where the product is evaluated, whose identity is also its overflow check.
    pub(super) overflow: Vec<(usize, Vec<usize>)>,

    pub(super) evaluated: Option<Vec<Block>>,

    /// The width the pure side computes its hints at, which no accumulator's bound reaches.
    pub(super) hint_bits: usize,
}

/// The schoolbook plan for operands whose limbs are bounded by `lhs` and `rhs`, plus an `addend`
/// with the given limb bounds (empty for none), at the answer's limb `widths`, on a field whose
/// widest injective width is `injective`.
///
/// The addend lets a division state `q·d + r` as one sum: limb `i` of `r` is one more
/// non-negative term in column `i`, so every argument below still holds.
///
/// [`None`] where one partial product and the headroom it is reduced in do not fit an element.
///
/// **Every bound is an integer bound, kept under the field.** A column sums non-negative partial
/// products and carries while the sum stays below `2^injective`, so the field element is the
/// integer sum. Before the next term would pass that, the column is reduced: its low limb is
/// range-checked at the limb width and the rest leaves as a carry range-checked at `carry_bits`.
/// The identity `low + 2^w·carry` is below `2^injective`, so it holds as integers too. A carry's
/// bound is its range check's, `2^carry_bits - 1`, not the honest maximum.
///
/// **The overflow check is on the operands.** All partial products are non-negative, so the
/// product is below `2^N` exactly when every partial product at or past the top column's place is
/// zero and the top column fits its width. The first half is `a_i · Σ b_j == 0` over the `b_j`
/// that land there with `a_i`; it holds as integers while the sum times the limb is below the
/// modulus, and a zero sum of non-negative terms is zero term by term. This costs one constraint
/// per left limb rather than one per partial product.
///
/// **Where summing costs more, the product is evaluated instead** (`evaluate`). Each partial
/// product of two witnessed limbs is a separate row, so the schoolbook pays about `k^2 / 2` rows
/// on two full operands, while evaluating pays `2k - 1` regardless. The caller counts the rows
/// summing would pay with [`witnessed_partial_products`] and evaluates only where both operands are
/// witnessed and that count is larger.
/// Every column's integer sum is non-negative, and the plan requires each to be below
/// `2^injective`, so the columns past the answer are zero as integers.
pub(super) fn plan_product(
    lhs: &[BigUint],
    rhs: &[BigUint],
    addend: &[BigUint],
    widths: &[usize],
    injective: usize,
    evaluate: bool,
) -> Option<ProductPlan> {
    let count = widths.len();
    let fits = |bound: &BigUint| bound.bits() <= injective as u64;
    let mut hint_bits = widths.iter().copied().max().unwrap_or(1);

    if evaluate {
        let (blocks, hint_bits) = plan_blocks(lhs, rhs, addend, widths, fits);
        return Some(ProductPlan {
            columns: Vec::new(),
            overflow: Vec::new(),
            evaluated: Some(blocks),
            hint_bits,
        });
    }

    let mut columns = Vec::with_capacity(count);
    let mut incoming: Vec<BigUint> = Vec::new();
    for (column, &width) in widths.iter().enumerate() {
        let top = column + 1 == count;
        let limit = BigUint::one() << width;

        let products: Vec<(Step, BigUint)> = (0..=column)
            .filter_map(|left| {
                let right = column - left;
                let bound = &lhs[left] * &rhs[right];
                (!bound.is_zero()).then_some((
                    Step::Product {
                        lhs: left,
                        rhs: right,
                    },
                    bound,
                ))
            })
            .collect();
        let added = addend
            .get(column)
            .filter(|bound| !bound.is_zero())
            .map(|bound| (Step::Addend(column), bound.clone()));
        let carries = std::mem::take(&mut incoming)
            .into_iter()
            .map(|bound| (Step::Carry, bound));
        let terms: Vec<(Step, BigUint)> =
            products.into_iter().chain(added).chain(carries).collect();

        let mut steps = Vec::new();
        let mut accumulated = BigUint::zero();
        let mut reduce = |accumulated: &mut BigUint, steps: &mut Vec<Step>| {
            if top {
                steps.push(Step::Reduce { carry_bits: None });
            } else {
                let carry_bits = (&*accumulated >> width).bits() as usize;
                steps.push(Step::Reduce {
                    carry_bits: Some(carry_bits),
                });
                incoming.push((BigUint::one() << carry_bits) - 1u8);
            }
            *accumulated = &limit - 1u8;
        };

        for (step, bound) in terms {
            if !fits(&(&accumulated + &bound)) {
                if accumulated < limit {
                    return None;
                }
                reduce(&mut accumulated, &mut steps);
                if !fits(&(&accumulated + &bound)) {
                    return None;
                }
            }
            accumulated += bound;
            hint_bits = hint_bits.max(accumulated.bits() as usize);
            steps.push(step);
        }
        if accumulated >= limit {
            reduce(&mut accumulated, &mut steps);
        }
        columns.push(steps);
    }

    let mut overflow = Vec::new();
    for (left, bound) in lhs.iter().enumerate() {
        if bound.is_zero() {
            continue;
        }
        let mut group = Vec::new();
        let mut sum = BigUint::zero();
        for right in count.saturating_sub(left)..count {
            if rhs[right].is_zero() {
                continue;
            }
            if !fits(&(bound * (&sum + &rhs[right]))) {
                if group.is_empty() {
                    return None;
                }
                overflow.push((left, std::mem::take(&mut group)));
                sum = BigUint::zero();
            }
            sum += &rhs[right];
            group.push(right);
        }
        if !group.is_empty() {
            overflow.push((left, group));
        }
    }

    Some(ProductPlan {
        columns,
        overflow,
        hint_bits,
        evaluated: None,
    })
}

fn plan_blocks(
    lhs_limb_bounds: &[BigUint],
    rhs_limb_bounds: &[BigUint],
    addend_bounds: &[BigUint],
    widths: &[usize],
    fits: impl Fn(&BigUint) -> bool,
) -> (Vec<Block>, usize) {
    let num_limbs = widths.len();
    if !(0..2 * num_limbs - 1)
        .all(|column| fits(&column_bound(lhs_limb_bounds, rhs_limb_bounds, column)))
    {
        ice!("the limb config admits a {num_limbs}-limb product with a column past the field");
    }

    let (term_bounds, carry_bounds, hint_bits) =
        calculate_bounds(lhs_limb_bounds, rhs_limb_bounds, addend_bounds, widths);

    let blocks = assign_blocks(&term_bounds, &carry_bounds, widths, fits);
    (blocks, hint_bits)
}

/// The integer bound of a column of the schoolbook.
fn column_bound(
    lhs_limb_bounds: &[BigUint],
    rhs_limb_bounds: &[BigUint],
    column_idx: usize,
) -> BigUint {
    (0..lhs_limb_bounds.len())
        .filter_map(|lhs_idx| {
            let rhs_idx = column_idx.checked_sub(lhs_idx)?;
            let rhs_limit = rhs_limb_bounds.get(rhs_idx)?;
            Some(&lhs_limb_bounds[lhs_idx] * rhs_limit)
        })
        .sum()
}

fn calculate_bounds(
    lhs_limb_bounds: &[BigUint],
    rhs_limb_bounds: &[BigUint],
    addend_bounds: &[BigUint],
    widths: &[usize],
) -> (Vec<BigUint>, Vec<BigUint>, usize) {
    let num_limbs = widths.len();
    let mut term_bounds = Vec::with_capacity(num_limbs);
    let mut carry_bounds = Vec::with_capacity(num_limbs + 1);
    let mut hint_bits = widths.iter().copied().max().unwrap_or(1);

    let mut carry = BigUint::zero();
    for (column, &width) in widths.iter().enumerate() {
        carry_bounds.push((BigUint::one() << carry.bits()) - 1u8);
        let term = column_bound(lhs_limb_bounds, rhs_limb_bounds, column)
            + addend_bounds.get(column).cloned().unwrap_or_default();
        let running = &term + &carry;
        hint_bits = hint_bits.max(running.bits() as usize);
        carry = &running >> width;
        term_bounds.push(term);
    }
    carry_bounds.push(BigUint::zero());

    (term_bounds, carry_bounds, hint_bits)
}

fn assign_blocks(
    term_bounds: &[BigUint],
    carry_bounds: &[BigUint],
    widths: &[usize],
    fits: impl Fn(&BigUint) -> bool,
) -> Vec<Block> {
    let num_limbs = widths.len();
    let mut blocks = Vec::new();
    let mut first = 0;
    let mut column = 0;
    while column < num_limbs {
        if fits(&calculate_block_bound(
            term_bounds,
            carry_bounds,
            widths,
            first,
            column,
        )) {
            column += 1;
        } else if column == first {
            ice!("column {column} of an evaluated product does not fit the field on its own");
        } else {
            blocks.push(Block {
                first,
                last: column - 1,
                carry_bits: Some(carry_bounds[column].bits() as usize),
            });
            first = column;
        }
    }
    blocks.push(Block {
        first,
        last: num_limbs - 1,
        carry_bits: None,
    });
    blocks
}

fn calculate_block_bound(
    terms: &[BigUint],
    carries: &[BigUint],
    widths: &[usize],
    first: usize,
    last: usize,
) -> BigUint {
    let mut summed = BigUint::zero();
    let mut place = BigUint::one();
    for column in first..=last {
        summed += &terms[column] * &place;
        place <<= widths[column];
    }
    let summed = summed + &carries[first];
    let reduced = &place * (&carries[last + 1] + 1u8) - 1u8;
    summed.max(reduced)
}

#[cfg(test)]
use mavros_artifacts::FieldConfig;
/// Whether [`plan_product`] plans a `bits`-wide schoolbook on `field` at full operand ranges, with
/// a full-range addend or without one.
#[cfg(test)]
pub(super) fn schoolbook_fits(field: FieldConfig, bits: usize, with_addend: bool) -> bool {
    use crate::compiler::passes::{
        shared::limbs::{widest_injective_int_bits, witness_limb_bits},
        wide_witness_ints::types::limb_widths,
    };

    let widths = limb_widths(bits, witness_limb_bits(field));
    let bounds = full_bounds(&widths);
    let addend = if with_addend { &bounds[..] } else { &[] };
    plan_product(
        &bounds,
        &bounds,
        addend,
        &widths,
        widest_injective_int_bits(field),
        true,
    )
    .is_some()
}

/// The bound of each limb at `widths` when nothing narrower is known: the largest value its width
/// holds.
pub(crate) fn full_bounds(widths: &[usize]) -> Vec<BigUint> {
    widths
        .iter()
        .map(|width| (BigUint::one() << *width) - 1u8)
        .collect()
}

/// How many products of two witnessed limbs summing `lhs · rhs` costs: one per partial product in
/// the answer, plus one per left limb for the overflow check.
///
/// Evaluating costs `2k - 1`, fewer whenever every limb is witnessed and `k >= 2`. A known limb
/// multiplies for free, so a widened narrow value is cheaper summed. A witnessed guard is left
/// out: it adds to both sides, and on two dense operands they tie at two limbs and evaluating wins
/// past that.
pub(super) fn witnessed_partial_products(lhs: &Factor, rhs: &Factor) -> usize {
    let count = lhs.limbs.len();
    let witnessed = |factor: &Factor, index: usize| {
        matches!(factor.limbs[index], Limb::Value(_)) && !factor.bounds[index].is_zero()
    };
    let mut products = 0;
    for left in (0..count).filter(|left| witnessed(lhs, *left)) {
        let right = (0..count).filter(|right| witnessed(rhs, *right));
        let (inside, past): (Vec<usize>, Vec<usize>) =
            right.partition(|right| left + right < count);
        products += inside.len() + usize::from(!past.is_empty());
    }
    products
}
