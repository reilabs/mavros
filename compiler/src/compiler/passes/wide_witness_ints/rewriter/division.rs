use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::shared::divmod_guard::nonzero_hint_divisor,
    ssa::{
        ValueId,
        hlssa::{BinaryArithOpKind, CastTarget, OpCode, builder::HLEmitter},
    },
};

use super::{
    super::types::limb_widths,
    CarryOut, Operands, Rewriter,
    product::Schoolbook,
    product::{Factor, Limb, past_the_single_cell_product},
    sign::Sign,
};
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;
use num_traits::{One, Zero};

/// What decides both answers of a division: dividend, divisor, guard, and whether it is signed.
pub(super) type Division = (ValueId, ValueId, Option<ValueId>, bool);

/// One answer of a division, as far as it has been built.
#[derive(Clone)]
pub(super) enum Answer {
    /// Limbs as the field elements [`Rewriter::deliver`] takes.
    Built(Vec<ValueId>),

    /// A signed answer's magnitude as limbs, each with whether it is witnessed, and its sign, which
    /// [`Rewriter::answer`] negates the magnitude by the first time it is asked for.
    Signed(Vec<(ValueId, bool)>, Sign),
}

// THE DIVISION
// ================================================================================================

impl Rewriter<'_> {
    /// An unsigned division or remainder whose product the single cell cannot hold; returns `true`
    /// if `op` was one and was lowered.
    ///
    /// The quotient `q` and remainder `r` are witnessed as limbs from a pure-side division and
    /// pinned by `q·d + r == n` and `r < d`. The first is one schoolbook, with `r` added to its
    /// columns and each column held to the dividend's limb; the second is the carry chain with its
    /// top borrow forced out, which a zero divisor cannot pass. Together they admit exactly the
    /// Euclidean pair.
    ///
    /// Under a guard, the pure side divides zero by one where the guard is off, so `q`, `r` and
    /// everything the schoolbook builds from them are zero there: its checks hold whatever the
    /// untaken branch left behind, and only `q·d + r == n` and `r < d` are guarded. The answer is
    /// those zeros, range-checked like any limb, so it needs no select. A value divided by itself,
    /// `d / d`, witnesses nothing: its answers are the constants one and zero, so where the guard
    /// is off they are selected to zero instead.
    pub(super) fn lower_division(&mut self, op: &OpCode) -> bool {
        let Some(Operands {
            bits,
            lhs: dividend,
            rhs: divisor,
        }) = divided(op, self.types, self.field)
        else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::BinaryArithOp { kind, result, .. } = inner else {
            ice_unreachable!("`divided` matches only a binary operation");
        };

        let widths = limb_widths(bits, self.limb_bits());
        let signed = kind.is_signed();
        let key = (dividend, divisor, guard, signed);
        let (mut quotient, mut remainder) = match self.divisions.remove(&key) {
            Some(answers) => answers,
            // `d / d` is one and `d % d` zero under either reading, wherever `d` is not zero.
            None if dividend == divisor => {
                let (q, r) = self.divide_by_itself(divisor, bits, &widths, guard);
                (Answer::Built(q), Answer::Built(r))
            }
            None if signed => self.divide_signed(dividend, divisor, bits, &widths, guard),
            None => {
                let (q, r) = self.divide(dividend, divisor, bits, &widths, guard);
                (Answer::Built(q), Answer::Built(r))
            }
        };
        let wanted = match kind {
            BinaryArithOpKind::UDiv | BinaryArithOpKind::SDiv => &mut quotient,
            BinaryArithOpKind::URem | BinaryArithOpKind::SRem => &mut remainder,
            _ => ice_unreachable!("`divided` matches only a division or a remainder"),
        };
        let answer = self.answer(wanted, &widths, guard);
        self.divisions.insert(key, (quotient, remainder));
        // A signed answer negates the magnitude by a chain checked only under the guard, so where
        // the guard is off the answer is selected to zero, as `d / d`'s is.
        let selected = if dividend == divisor || signed {
            guard
        } else {
            None
        };
        self.deliver(*result, answer, &widths, bits, selected);
        true
    }

    /// `answer`'s limbs as the field elements [`Self::deliver`] takes, negating a signed one by
    /// its sign under `guard` and keeping what that builds for the next time it is asked for.
    fn answer(
        &mut self,
        answer: &mut Answer,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> Vec<ValueId> {
        match answer {
            Answer::Built(limbs) => limbs.clone(),
            Answer::Signed(magnitude, sign) => {
                let negated = self.negate_if(magnitude.clone(), widths, *sign, guard);
                let limbs = self.limb_fields(negated);
                *answer = Answer::Built(limbs.clone());
                limbs
            }
        }
    }

    /// The quotient and the remainder of `dividend / divisor`, each as its limbs' field elements.
    fn divide(
        &mut self,
        dividend: ValueId,
        divisor: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, Vec<ValueId>) {
        let n = self.factor(dividend, bits, widths);
        let d = self.factor(divisor, bits, widths);
        let pure_dividend = self.pure_whole(dividend, bits);
        let pure_divisor = self.pure_whole(divisor, bits);
        let (q, r) = self.divide_factors(&n, &d, pure_dividend, pure_divisor, bits, widths, guard);
        let quotient = q.limbs.iter().map(|limb| self.limb_field(limb)).collect();
        let remainder = r.limbs.iter().map(|limb| self.limb_field(limb)).collect();
        (quotient, remainder)
    }

    /// The quotient and the remainder of `n / d` as limbs, witnessed from the pure side's
    /// `pure_dividend / pure_divisor`, the operands' whole values there.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn divide_factors(
        &mut self,
        n: &Factor,
        d: &Factor,
        mut pure_dividend: ValueId,
        mut pure_divisor: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Factor, Factor) {
        // Bound the answers by `q <= n / d` and `r < d`; only known limbs, and most of all a
        // constant divisor, tighten the width's bound. The chain below refuses a zero divisor, so
        // the bounds need only hold for `d >= 1`.
        let limb_bits = self.limb_bits();
        let dividend_max = n.largest(limb_bits);
        let divisor_max = d.largest(limb_bits);
        let divisor_min = d.smallest(limb_bits).max(BigUint::one());
        let quotient_max = &dividend_max / divisor_min;
        let remainder_max = if divisor_max.is_zero() {
            BigUint::zero()
        } else {
            (&divisor_max - 1u8).min(dividend_max)
        };

        if let Some(condition) = guard {
            let condition = self.pure_of(condition, self.is_witness(condition));
            let zero = self.int_const(IntBits::zero(bits));
            let one = self.int_const(IntBits::one(bits));
            pure_dividend = self.select(condition, pure_dividend, zero);
            pure_divisor = self.select(condition, pure_divisor, one);
        }
        let pure_divisor = nonzero_hint_divisor(self, pure_divisor, bits);
        let quotient_hint = self.bin(BinaryArithOpKind::UDiv, pure_dividend, pure_divisor);
        let remainder_hint = self.bin(BinaryArithOpKind::URem, pure_dividend, pure_divisor);
        let q = self.witness_hinted(quotient_hint, bits, widths, &quotient_max);
        let r = self.witness_hinted(remainder_hint, bits, widths, &remainder_max);

        let target: Vec<ValueId> = n.limbs.iter().map(|limb| self.limb_field(limb)).collect();
        self.schoolbook(
            Schoolbook {
                lhs: &q,
                rhs: d,
                addend: Some(&r),
                pinned: Some(&target),
                widths,
                bits,
                operation: "division",
            },
            guard,
        );

        // `r < d` reads only the limbs either side can reach, as above them both are zero.
        let reached = (0..widths.len())
            .rev()
            .find(|index| !r.bounds[*index].is_zero() || !d.bounds[*index].is_zero())
            .map_or(1, |top| top + 1);
        let remainder_limbs = self.chain_limbs_of(&r, reached);
        let divisor_limbs = self.chain_limbs_of(d, reached);
        self.carry_chain_over(
            true,
            &remainder_limbs,
            &divisor_limbs,
            widths[..reached].to_vec(),
            CarryOut::One,
            guard,
        );
        (q, r)
    }

    /// `d / d` and `d % d`, which are one and zero wherever `d` is not zero, so only that is
    /// checked: `0 < d`, the chain with its top borrow forced out.
    fn divide_by_itself(
        &mut self,
        divisor: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, Vec<ValueId>) {
        let divisor = self.chain_limbs(divisor, bits, widths.len());
        let zero: Vec<(ValueId, bool)> = widths
            .iter()
            .map(|width| (self.int_const(IntBits::zero(*width)), false))
            .collect();
        self.carry_chain_over(true, &zero, &divisor, widths.to_vec(), CarryOut::One, guard);

        let (zero, one) = (
            self.field_const(self.field.zero()),
            self.field_const(self.field.one()),
        );
        let mut quotient = vec![zero; widths.len()];
        quotient[0] = one;
        (quotient, vec![zero; widths.len()])
    }

    /// A pure `bits`-wide `hint` witnessed as limbs at `widths`, each range-checked no wider than
    /// `max` allows for it. A limb `max` does not reach is a constant zero, which drops out of the
    /// schoolbook's plan. Only the caller's constraints tie the limbs to anything.
    fn witness_hinted(
        &mut self,
        hint: ValueId,
        bits: usize,
        widths: &[usize],
        max: &BigUint,
    ) -> Factor {
        let mut limbs = Vec::with_capacity(widths.len());
        let mut bounds = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let low = index * self.limb_bits();
            let checked = ((max >> low).bits() as usize).min(*width);
            if checked == 0 {
                limbs.push(Limb::Constant(IntBits::zero(*width)));
                bounds.push(BigUint::zero());
                continue;
            }

            let (limb, _) = self.witness_limb(hint, bits, index, *width, checked);
            limbs.push(Limb::Value(limb));
            bounds.push((BigUint::one() << checked) - 1u8);
        }
        Factor {
            limbs,
            witnessed: true,
            bounds,
        }
    }

    /// A value as one pure integer of `bits`, which is what the pure side divides.
    pub(super) fn pure_whole(&mut self, value: ValueId, bits: usize) -> ValueId {
        if !self.is_witness(value) {
            return value;
        }
        match self.wide_width(value) {
            Some(_) => {
                let limbs = self.pure_limbs_at(value, bits, bits);
                self.recombine_pure(&limbs, bits)
            }
            None => self.cast(value, CastTarget::ValueOf),
        }
    }

    /// A limb of a [`Factor`] as a field element.
    pub(super) fn limb_field(&mut self, limb: &Limb) -> ValueId {
        match limb {
            Limb::Value(limb) => self.cast(*limb, CastTarget::Field),
            // A witness limb is never wider than the host word, so its pattern is one host limb.
            Limb::Constant(pattern) => self.field_const(self.field.constant(pattern.limbs()[0])),
        }
    }

    /// The low `count` limbs of a [`Factor`] as the carry chain reads them.
    pub(super) fn chain_limbs_of(&mut self, factor: &Factor, count: usize) -> Vec<(ValueId, bool)> {
        factor.limbs[..count]
            .iter()
            .map(|limb| match limb {
                Limb::Value(limb) => (*limb, factor.witnessed),
                Limb::Constant(pattern) => (self.int_const(pattern.clone()), false),
            })
            .collect()
    }
}

/// The division the representation lowers, if `op` is one: a division or remainder under either
/// reading, guarded or not, with a witnessed operand, past [`single_cell_product_fits`], where the
/// single-cell lowering's `q·d + r` could wrap the modulus.
pub(super) fn divided(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
) -> Option<Operands> {
    past_the_single_cell_product(
        op,
        types,
        field,
        &[
            BinaryArithOpKind::UDiv,
            BinaryArithOpKind::URem,
            BinaryArithOpKind::SDiv,
            BinaryArithOpKind::SRem,
        ],
    )
}
