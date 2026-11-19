use super::identity::{
    Factor, IdentityShape, IdentityTemplate, Limb, Polarity, PolynomialIdentity, Target, Term,
};
use super::{
    Operands, Rewriter, divided,
    division::EuclideanDivision,
    lower::{unguarded, witnessed_width},
    sign::SignedProduct,
};
use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::{
        shared::limbs::{
            single_cell_product_fits, single_cell_signed_product_fits, widest_injective_int_bits,
        },
        wide_witness_ints::types::limb_widths,
    },
    ssa::{
        ValueId,
        hlssa::{BinaryArithOpKind, CastTarget, Constant, OpCode, builder::HLEmitter},
    },
};
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;
use num_traits::One;
#[cfg(test)]
pub(crate) use tests::division_carries;
#[cfg(test)]
mod tests;
#[derive(Clone, Copy)]
struct IdentityFlags {
    range_check_flag: Option<ValueId>,

    answer_flag: Option<ValueId>,
}

impl IdentityFlags {
    fn new(
        rewriter: &mut Rewriter<'_>,
        guard: Option<ValueId>,
        target: Option<Target<'_>>,
    ) -> Self {
        Self {
            range_check_flag: guard.filter(|_| !matches!(target, Some(Target::Given(_)))),
            answer_flag: guard.map(|condition| rewriter.cast(condition, CastTarget::Field)),
        }
    }

    /// The range check guard on the pure side at `hint`.
    fn pure_range_check_guard(
        &self,
        rewriter: &mut Rewriter<'_>,
        hint: CastTarget,
    ) -> Option<ValueId> {
        self.range_check_flag.map(|condition| {
            let pure = rewriter.pure_of(condition, rewriter.is_witness(condition));
            rewriter.cast(pure, hint)
        })
    }
}

impl Rewriter<'_> {
    /// An unsigned product the single cell cannot hold. Returns whether `op` was one and was
    /// lowered.
    ///
    /// A schoolbook at the witness limb, checked by the polynomial identity.
    pub(super) fn lower_product(&mut self, op: &OpCode) -> bool {
        let Some(Operands { bits, lhs, rhs }) = multiplied(op, self.types, self.field()) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::BinaryArithOp { kind, result, .. } = inner else {
            ice_unreachable!("`multiplied` matches only a binary operation");
        };
        if *kind == BinaryArithOpKind::SMul {
            self.lower_signed_product(*result, lhs, rhs, bits, guard);
            return true;
        }

        let widths = limb_widths(bits, self.limb_bits());
        let lhs = self.factor(lhs, bits, &widths);
        let rhs = self.factor(rhs, bits, &widths);
        let identity = self.identity(UnsignedProduct, &widths, (&lhs, &rhs, &widths));
        let answer = self.evaluate_f_poly(&identity, guard);
        self.deliver(*result, answer, &widths, bits, guard);
        true
    }

    /// The identity `shape` lays out over `factors` at limb `widths`.
    ///
    /// [`LimbConfig`](super::super::limb_config::LimbConfig) admitted every identity the program
    /// lowers at this limb, so the field holds it.
    pub(super) fn identity<'f, S: IdentityTemplate<'f>>(
        &self,
        shape: S,
        widths: &[usize],
        factors: S::Factors,
    ) -> PolynomialIdentity<'f> {
        shape
            .instantiate(widths, widest_injective_int_bits(self.field()), factors)
            .unwrap_or_else(|| {
            ice_unreachable!(
                "the limb config admits every identity it lowers, but not {shape:?} at {widths:?}"
            )
        })
    }

    /// An operand of the schoolbook as its limbs at `bits`, with their bounds.
    ///
    /// A constant is cut at compile time, so limbs it does not reach are known zeros and drop out
    /// of the plan. A witnessed operand can have known limbs too (above a widened value, emptied by
    /// a constant shift, or known from an earlier gadget); each is taken as its constant, which is
    /// pinned by being one.
    pub(super) fn factor(&mut self, value: ValueId, bits: usize, widths: &[usize]) -> Factor {
        if let Some(constant) = self.ssa.get_const(value)
            && let Constant::Int(pattern) = constant.as_ref()
        {
            let limbs = widths
                .iter()
                .enumerate()
                .map(|(index, width)| {
                    Limb::Constant(pattern.bit_range(index * self.limb_bits(), *width))
                })
                .collect();
            return Factor::new(limbs, widths, false, None);
        }

        let limbs = self
            .chain_operand(value, bits, widths.len())
            .into_iter()
            .zip(widths)
            .map(|(limb, width)| match self.known(limb) {
                Some(pattern) => Limb::Constant(pattern.cast(*width)),
                None => Limb::Value(limb),
            })
            .collect();
        Factor::new(limbs, widths, self.is_witness(value), None)
    }

    /// Witnesses and checks `f_polynomial`,
    /// Returning the answer's limbs where it witnesses them.
    pub(super) fn evaluate_f_poly(
        &mut self,
        f_polynomial: &PolynomialIdentity<'_>,
        guard: Option<ValueId>,
    ) -> Vec<ValueId> {
        let terms = &f_polynomial.terms[..];
        let pure_integer_type = CastTarget::Int(f_polynomial.hint_bits());
        let guards = IdentityFlags::new(self, guard, f_polynomial.target);
        let pure_range_check_guard = guards.pure_range_check_guard(self, pure_integer_type);

        let (witnessed_answer, carries) = self.witness_columns(
            f_polynomial,
            guards.range_check_flag,
            pure_range_check_guard,
        );

        let answer_as_felts =
            self.ansswer_as_fields(f_polynomial.target, &witnessed_answer, guards.answer_flag);

        self.check_block_carries(f_polynomial, &carries, guards.range_check_flag);
        let terms_as_felts = self.terms_as_fields(&f_polynomial, guards.answer_flag);

        for point in 0..=f_polynomial.degree() as u64 {
            self.check_at_point(
                point,
                terms,
                &terms_as_felts,
                &carries,
                &answer_as_felts,
                pure_range_check_guard,
            );
        }
        witnessed_answer
    }

    /// What the polynomial's carry chain witnesses:
    /// the answer's limbs, where they are solved, and each carry.
    fn witness_columns(
        &mut self,
        f_polynomial: &PolynomialIdentity<'_>,
        range_check_guard: Option<ValueId>,
        pure_range_check_guard: Option<ValueId>,
    ) -> (Vec<ValueId>, Vec<ValueId>) {
        let target = f_polynomial.target;
        let hint_bits = f_polynomial.hint_bits();

        let mut witnessed_answer = Vec::new();
        let mut carries = Vec::with_capacity(f_polynomial.num_committed_carries());

        let mut carry_in: Option<ValueId> = None;
        for column in 0..f_polynomial.carry_chain_len() {
            let mut running_sum = self.pure_column(&f_polynomial.terms, column, hint_bits);
            if let Some(mask) = pure_range_check_guard {
                running_sum = self.bin(BinaryArithOpKind::SMul, mask, running_sum);
            }
            if let Some(carry) = carry_in {
                running_sum = self.bin(BinaryArithOpKind::SAdd, running_sum, carry);
            }
            if let Some(Target::Witnessed(widths)) = target
                && column < widths.len()
            {
                let low = self.cast(running_sum, CastTarget::Int(widths[column]));
                let low = self.cast(low, CastTarget::Field);
                let written = self.write_witness(low);
                self.push_guarded(
                    range_check_guard,
                    OpCode::Rangecheck {
                        value: written,
                        max_bits: widths[column],
                    },
                );
                witnessed_answer.push(written);
            }
            if column >= f_polynomial.num_committed_carries() {
                continue;
            }
            let (carry, witnessed_carry) = self.witness_carry(f_polynomial, running_sum);
            carries.push(witnessed_carry);
            carry_in = Some(carry);
        }
        (witnessed_answer, carries)
    }

    /// The carry out of a column whose running sum is `running`, as its hint and its witness.
    fn witness_carry(
        &mut self,
        identity: &PolynomialIdentity<'_>,
        running: ValueId,
    ) -> (ValueId, ValueId) {
        match identity.carry_offset_bits() {
            None => {
                let carry = self.shifted_down(running, identity.hint_bits(), self.limb_bits());
                let carry_as_field = self.cast(carry, CastTarget::Field);
                (carry, self.write_witness(carry_as_field))
            }
            Some(offset_bits) => {
                let limb_bits = self.int_const(IntBits::from_u128(
                    identity.hint_bits(),
                    self.limb_bits() as u128,
                ));
                let carry = self.bin(BinaryArithOpKind::SShr, running, limb_bits);
                let offset = self.int_const(IntBits::from_biguint(
                    identity.hint_bits(),
                    &(BigUint::one() << offset_bits),
                ));
                let raised = self.bin(BinaryArithOpKind::SAdd, carry, offset);
                let raised = self.cast(raised, CastTarget::Field);
                let field_offset = self.field_const(self.field().two_pow(offset_bits));
                let lowered = self.bin(BinaryArithOpKind::USub, raised, field_offset);
                (carry, self.write_witness(lowered))
            }
        }
    }

    /// Range check the carry out of each block.
    fn check_block_carries(
        &mut self,
        identity: &PolynomialIdentity<'_>,
        carries: &[ValueId],
        check: Option<ValueId>,
    ) {
        let offset = identity
            .carry_offset_bits()
            .map(|offset_bits| self.field_const(self.field().two_pow(offset_bits)));
        for block in identity.blocks() {
            if let Some(carry_bits) = block.carry_bits {
                let carry = carries[block.last];
                let value = match offset {
                    Some(offset) => self.bin(BinaryArithOpKind::UAdd, carry, offset),
                    None => carry,
                };
                self.push_guarded(
                    check,
                    OpCode::Rangecheck {
                        value,
                        max_bits: carry_bits,
                    },
                );
            }
        }
    }

    /// The limbs of each term's factors as field elements, a gated term's left factor scaled by `flag`.
    fn terms_as_fields(
        &mut self,
        f_polynomial: &PolynomialIdentity<'_>,
        flag: Option<ValueId>,
    ) -> Vec<(Vec<ValueId>, Option<Vec<ValueId>>)> {
        let terms = &f_polynomial.terms[..];
        let mut fields: Vec<(Vec<ValueId>, Option<Vec<ValueId>>)> = Vec::with_capacity(terms.len());
        for term in terms {
            let lhs = (0..term.lhs.limbs.len())
                .map(|limb| self.factor_limb_constrained(term.lhs, limb))
                .collect();
            let rhs = term.rhs.map(|rhs| {
                (0..rhs.limbs.len())
                    .map(|limb| self.factor_limb_constrained(rhs, limb))
                    .collect()
            });
            fields.push((lhs, rhs));
        }

        if let Some(flag) = flag {
            for (term, (lhs, _)) in terms.iter().zip(&mut fields) {
                if term.gated {
                    for limb in lhs {
                        *limb = self.bin(BinaryArithOpKind::UMul, flag, *limb);
                    }
                }
            }
        }
        fields
    }

    /// The answer as field_elements
    fn ansswer_as_fields(
        &mut self,
        target: Option<Target<'_>>,
        witnessed_answer_limbs: &[ValueId],
        flag: Option<ValueId>,
    ) -> Vec<ValueId> {
        match target {
            // No specific answer term in polynomial.
            None => Vec::new(),
            // Answer has been solved in previous stage: return.
            Some(Target::Witnessed(_)) => witnessed_answer_limbs.to_vec(),
            // Answer limbs were supplied as witnesses. Mask if needed.
            Some(Target::Given(given)) => given
                .limbs
                .iter()
                .map(|limb| {
                    let limb = self.limb_field(limb);
                    match flag {
                        Some(flag) => self.bin(BinaryArithOpKind::UMul, flag, limb),
                        None => limb,
                    }
                })
                .collect(),
        }
    }

    /// The identity's row at `point`: `Σ ±terms − (β − point)·H − C = 0`, with one product of
    /// witnesses as the row's multiplication and each further one witnessed and pinned by a row of
    /// its own.
    fn check_at_point(
        &mut self,
        point: u64,
        terms: &[Term<'_>],
        terms_as_felts: &[(Vec<ValueId>, Option<Vec<ValueId>>)],
        carries: &[ValueId],
        answer: &[ValueId],
        mask: Option<ValueId>,
    ) {
        let radix = self.field().two_pow(self.limb_bits());

        let mut quadratic = Vec::new();
        let mut rest: Vec<(Polarity, ValueId)> = Vec::new();
        for (term, (lhs, rhs)) in terms.iter().zip(terms_as_felts) {
            let a = self.evaluate_at(lhs, point);
            match rhs {
                Some(rhs) => {
                    let b = self.evaluate_at(rhs, point);
                    match term
                        .rhs
                        .filter(|rhs| term.lhs.is_witnessed && rhs.is_witnessed)
                    {
                        Some(rhs) => quadratic.push((term, rhs, a, b)),
                        None => {
                            let product = self.bin(BinaryArithOpKind::UMul, a, b);
                            rest.push((term.polarity, product));
                        }
                    }
                }
                None => rest.push((term.polarity, a)),
            }
        }

        if !carries.is_empty() {
            let h = self.evaluate_at(carries, point);
            let scale = self.field_const(radix - self.field().constant(point));
            let h = self.bin(BinaryArithOpKind::UMul, h, scale);
            rest.push((Polarity::Negative, h));
        }

        if !answer.is_empty() {
            let c = self.evaluate_at(answer, point);
            rest.push((Polarity::Negative, c));
        }

        let head = quadratic
            .first()
            .map(|(term, _, a, b)| (term.polarity, *a, *b));

        for (term, rhs, a, b) in quadratic.into_iter().skip(1) {
            // A second product of witnesses has no place in the row: it is witnessed from the pure
            // side and pinned by a row of its own.
            let mask = mask.filter(|_| term.gated);
            let written = self.witness_product_at(point, term.lhs, rhs, mask);
            self.push(OpCode::Constrain { a, b, c: written });
            rest.push((term.polarity, written));
        }

        match head {
            None => {
                let sum = self.signed_sum(&rest, &CastTarget::Field);
                let zero = self.field_const(self.field().zero());
                self.constrain_equal(sum, zero, None);
            }
            Some((polarity, a, b)) => {
                if polarity == Polarity::Positive {
                    for (polarity, _) in &mut rest {
                        *polarity = polarity.flipped();
                    }
                }
                let c = self.signed_sum(&rest, &CastTarget::Field);
                self.push(OpCode::Constrain { a, b, c });
            }
        }
    }

    /// The product of factors `lhs` and `rhs` at `point`, witnessed from the pure side, the left
    /// one scaled by `mask` where there is one.
    fn witness_product_at(
        &mut self,
        point: u64,
        lhs: &Factor,
        rhs: &Factor,
        mask: Option<ValueId>,
    ) -> ValueId {
        let pure_lhs: Vec<_> = (0..lhs.limbs.len())
            .map(|limb| self.factor_limb_pure(lhs, limb, &CastTarget::Field))
            .collect();
        let pure_rhs: Vec<_> = (0..rhs.limbs.len())
            .map(|limb| self.factor_limb_pure(rhs, limb, &CastTarget::Field))
            .collect();
        let mut pa = self.evaluate_at(&pure_lhs, point);
        if let Some(mask) = mask {
            let mask = self.cast(mask, CastTarget::Field);
            pa = self.bin(BinaryArithOpKind::UMul, mask, pa);
        }
        let pb = self.evaluate_at(&pure_rhs, point);
        let product = self.bin(BinaryArithOpKind::UMul, pa, pb);
        self.write_witness(product)
    }

    /// `Σ ±value` as one value of type `target`. The subtracted values come last, so the sum starts
    /// from an added one wherever there is one.
    fn signed_sum(&mut self, terms: &[(Polarity, ValueId)], target: &CastTarget) -> ValueId {
        let zero = |rewriter: &mut Self| match target {
            CastTarget::Field => rewriter.field_const(rewriter.field().zero()),
            CastTarget::Int(bits) => rewriter.int_const(IntBits::zero(*bits)),
            _ => ice_unreachable!("a signed sum is of a field element or an integer"),
        };
        let mut sum: Option<ValueId> = None;
        for wanted in [Polarity::Positive, Polarity::Negative] {
            for (polarity, value) in terms {
                if *polarity != wanted {
                    continue;
                }
                sum = Some(match (sum, wanted) {
                    (None, Polarity::Positive) => *value,
                    (None, Polarity::Negative) => {
                        let zero = zero(self);
                        self.bin(BinaryArithOpKind::USub, zero, *value)
                    }
                    (Some(sum), Polarity::Positive) => {
                        self.bin(BinaryArithOpKind::UAdd, sum, *value)
                    }
                    (Some(sum), Polarity::Negative) => {
                        self.bin(BinaryArithOpKind::USub, sum, *value)
                    }
                });
            }
        }
        sum.unwrap_or_else(|| zero(self))
    }

    /// Calculate column `column` of `terms`.
    fn pure_column(&mut self, terms: &[Term<'_>], column_idx: usize, hint_bits: usize) -> ValueId {
        let hint_type = CastTarget::Int(hint_bits);
        let mut values: Vec<(Polarity, ValueId)> = Vec::new();
        for term in terms {
            match term.rhs {
                Some(rhs_factor) => {
                    for lhs_limb in 0..term.lhs.limbs.len() {
                        let Some(rhs_limb) = column_idx.checked_sub(lhs_limb) else {
                            continue;
                        };
                        if rhs_limb >= rhs_factor.limbs.len()
                            || term.lhs.vanishes(lhs_limb)
                            || rhs_factor.vanishes(rhs_limb)
                        {
                            continue;
                        }
                        let a = self.factor_limb_pure(term.lhs, lhs_limb, &hint_type);
                        let b = self.factor_limb_pure(rhs_factor, rhs_limb, &hint_type);
                        let product = self.bin(BinaryArithOpKind::UMul, a, b);
                        values.push((term.polarity, product));
                    }
                }
                None => {
                    if column_idx < term.lhs.limbs.len() && !term.lhs.vanishes(column_idx) {
                        let value = self.factor_limb_pure(term.lhs, column_idx, &hint_type);
                        values.push((term.polarity, value));
                    }
                }
            }
        }
        self.signed_sum(&values, &hint_type)
    }

    /// `Σ coefficients[i] · point^i` as a field element, which is linear in the coefficients.
    fn evaluate_at(&mut self, coefficients: &[ValueId], point: u64) -> ValueId {
        if point == 0 {
            return coefficients[0];
        }
        let base = self.field().constant(point);
        let mut power = self.field().one();
        let mut sum = None;
        for coefficient in coefficients {
            let term = if power == self.field().one() {
                *coefficient
            } else {
                let place = self.field_const(power);
                self.bin(BinaryArithOpKind::UMul, *coefficient, place)
            };
            sum = Some(match sum {
                None => term,
                Some(sum) => self.bin(BinaryArithOpKind::UAdd, sum, term),
            });
            power = power * base;
        }
        sum.expect("a polynomial has at least one coefficient")
    }

    /// `lhs == rhs` as field elements, where `guard` holds.
    pub(super) fn constrain_equal(&mut self, lhs: ValueId, rhs: ValueId, guard: Option<ValueId>) {
        let difference = self.bin(BinaryArithOpKind::USub, lhs, rhs);
        let flag = match guard {
            Some(condition) => self.cast(condition, CastTarget::Field),
            None => self.field_const(self.field().one()),
        };
        let zero = self.field_const(self.field().zero());
        self.push(OpCode::Constrain {
            a: flag,
            b: difference,
            c: zero,
        });
    }

    /// Limb `limb` of `factor` on the constraining side.
    fn factor_limb_constrained(&mut self, factor: &Factor, limb: usize) -> ValueId {
        let value = self.limb_field(&factor.limbs[limb]);
        self.sub_offset(factor, limb, value, &CastTarget::Field)
    }

    /// Limb `limb` of `factor`'s pure value at the `hint_type`.
    fn factor_limb_pure(
        &mut self,
        factor: &Factor,
        limb: usize,
        hint_type: &CastTarget,
    ) -> ValueId {
        let value = match &factor.limbs[limb] {
            Limb::Value(value) => {
                let pure = self.pure_of(*value, factor.is_witnessed);
                self.cast(pure, hint_type.clone())
            }
            Limb::Constant(pattern) => match hint_type {
                CastTarget::Field => self.limb_const(pattern),
                CastTarget::Int(bits) => self.int_const(pattern.cast(*bits)),
                _ => ice_unreachable!("a pure limb is a field element or an integer"),
            },
        };
        self.sub_offset(factor, limb, value, hint_type)
    }

    /// `value`, limb `limb` of `factor` as `target`, less the limb's offset where it has one.
    fn sub_offset(
        &mut self,
        factor: &Factor,
        limb: usize,
        value: ValueId,
        target: &CastTarget,
    ) -> ValueId {
        let Some(offset) = factor.top_offset(limb) else {
            return value;
        };
        match target {
            CastTarget::Field => {
                let constant =
                    self.limb_const(&IntBits::from_biguint(offset.bits() as usize, offset));
                self.bin(BinaryArithOpKind::USub, value, constant)
            }
            CastTarget::Int(bits) => {
                let modulus = BigUint::one() << bits;
                let wrapped = (&modulus - offset % &modulus) % &modulus;
                let constant = self.int_const(IntBits::from_biguint(*bits, &wrapped));
                self.bin(BinaryArithOpKind::UAdd, value, constant)
            }
            _ => ice_unreachable!("a limb's offset is taken in a field element or an integer"),
        }
    }
}

/// `a·b = c`, solving for `c`.
#[derive(Clone, Copy, Debug)]
pub(super) struct UnsignedProduct;

impl IdentityShape for UnsignedProduct {
    fn num_products(&self) -> usize {
        1
    }

    fn num_linear_terms(&self) -> usize {
        0
    }

    fn is_checked(&self) -> bool {
        true
    }
}

impl<'a> IdentityTemplate<'a> for UnsignedProduct {
    /// `a`, `b`, and the widths `c` is solved at.
    type Factors = (&'a Factor, &'a Factor, &'a [usize]);

    fn terms(self, (a, b, widths): Self::Factors) -> (Vec<Term<'a>>, Option<Target<'a>>) {
        let product = Term {
            polarity: Polarity::Positive,
            lhs: a,
            rhs: Some(b),
            gated: true,
        };
        (vec![product], Some(Target::Witnessed(widths)))
    }
}

/// The identity `op` lowers to, if it lowers to one, at the width of its operands.
///
/// A signed division divides the magnitudes, so it is the unsigned division's identity.
pub(in super::super) fn lowered_identity(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
) -> Option<(usize, &'static dyn IdentityShape)> {
    let Operands { bits, .. } =
        multiplied(op, types, field).or_else(|| divided(op, types, field))?;
    let OpCode::BinaryArithOp { kind, .. } = unguarded(op) else {
        ice_unreachable!("`multiplied` and `divided` match only a binary operation");
    };
    let shape: &'static dyn IdentityShape = match kind {
        BinaryArithOpKind::UMul => &UnsignedProduct,
        BinaryArithOpKind::SMul => &SignedProduct,
        BinaryArithOpKind::UDiv
        | BinaryArithOpKind::URem
        | BinaryArithOpKind::SDiv
        | BinaryArithOpKind::SRem => &EuclideanDivision,
        _ => ice_unreachable!("`multiplied` and `divided` match only products and divisions"),
    };
    Some((bits, shape))
}

/// The product the schoolbook lowers, if `op` is one: a multiplication under either reading,
/// guarded or not, with a witnessed operand, at a width the single cell does not take.
pub(in super::super) fn multiplied(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
) -> Option<Operands> {
    past_the_single_cell_product(
        op,
        types,
        field,
        &[BinaryArithOpKind::UMul, BinaryArithOpKind::SMul],
    )
}

/// A binary operation of one of `kinds`, guarded or not, with a witnessed operand, at a width whose
/// product one element cannot hold: past [`single_cell_product_fits`] for an unsigned one, and
/// past [`single_cell_signed_product_fits`] for a signed one.
pub(super) fn past_the_single_cell_product(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
    kinds: &[BinaryArithOpKind],
) -> Option<Operands> {
    let OpCode::BinaryArithOp { kind, lhs, rhs, .. } = unguarded(op) else {
        return None;
    };
    if !kinds.contains(kind) {
        return None;
    }
    let bits = witnessed_width(types, *lhs, *rhs)?;
    let fits = if kind.is_signed() {
        single_cell_signed_product_fits(field, bits)
    } else {
        single_cell_product_fits(field, bits)
    };
    (!fits).then_some(Operands {
        bits,
        lhs: *lhs,
        rhs: *rhs,
    })
}
