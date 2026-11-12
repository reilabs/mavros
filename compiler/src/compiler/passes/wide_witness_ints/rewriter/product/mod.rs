use super::{
    Operands, Rewriter,
    lower::{unguarded, witnessed_width},
};
use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::{
        shared::{
            limbs::{
                single_cell_product_fits, single_cell_signed_product_fits,
                widest_injective_int_bits,
            },
            unsupported::unsupported_on_this_field,
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
use plan::{Block, ProductPlan, Step, plan_product, witnessed_partial_products};
pub(super) use plan::{Factor, Limb, full_bounds};
#[cfg(test)]
pub(crate) use tests::division_carries;
mod plan;
#[cfg(test)]
mod tests;
/// A column being accumulated: its field element and its mirror on the pure side, or nothing while
/// it is still zero.
#[derive(Clone, Copy)]
struct Accumulator {
    sum: Option<(ValueId, ValueId)>,
}

/// One schoolbook: `lhs · rhs`, plus `addend` where there is one, either range-checked into limbs
/// or `pinned` to a target's.
#[derive(Clone, Copy)]
pub(super) struct Schoolbook<'a> {
    pub(super) lhs: &'a Factor,
    pub(super) rhs: &'a Factor,
    pub(super) addend: Option<&'a Factor>,

    /// The target's limbs as field elements, each already bounded at its width.
    pub(super) pinned: Option<&'a [ValueId]>,

    /// The answer's limb widths, which are the operands' too.
    pub(super) widths: &'a [usize],

    /// The width the operation is at, for a refusal.
    pub(super) bits: usize,

    /// What the operation is called, for a refusal.
    pub(super) operation: &'static str,

    pub(super) guard: Option<ValueId>,
}

#[derive(Clone, Copy)]
struct Guards {
    /// The range and overflow checks.
    check: Option<ValueId>,
    /// The equalities to a pinned target.
    pin: Option<ValueId>,
}

impl Schoolbook<'_> {
    fn guards(&self) -> Guards {
        Guards {
            check: if self.pinned.is_some() {
                None
            } else {
                self.guard
            },
            pin: self.guard,
        }
    }
}

impl Rewriter<'_> {
    /// An unsigned product the single cell cannot hold. Returns whether `op` was one and was
    /// lowered.
    ///
    /// A schoolbook at the witness limb, following [`plan_product`], which states the soundness
    /// argument. Under a guard the range and overflow checks are off where the guard is off, and
    /// the answer is zero there; the carries are written either way.
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
        let answer = self.schoolbook(Schoolbook {
            lhs: &lhs,
            rhs: &rhs,
            addend: None,
            pinned: None,
            widths: &widths,
            bits,
            operation: "multiplication",
            guard,
        });
        self.deliver(*result, answer, &widths, bits, guard);
        true
    }

    /// `lhs · rhs (+ addend)` at `widths`, following [`plan_product`], as the field element of each
    /// answer limb.
    pub(super) fn schoolbook(&mut self, product: Schoolbook<'_>) -> Vec<ValueId> {
        let Schoolbook {
            lhs,
            rhs,
            addend,
            widths,
            bits,
            operation,
            ..
        } = product;
        let evaluate = lhs.witnessed
            && rhs.witnessed
            && witnessed_partial_products(lhs, rhs) > 2 * widths.len() - 1;
        let plan = plan_product(
            &lhs.bounds,
            &rhs.bounds,
            addend.map_or(&[][..], |addend| &addend.bounds),
            widths,
            widest_injective_int_bits(self.field()),
            evaluate,
        )
        .unwrap_or_else(|| {
            unsupported_on_this_field(
                format_args!(
                    "a {bits}-bit unsigned {operation} is a schoolbook product of witness limbs, which needs one partial product and the carry it is reduced into to fit a field element"
                ),
                self.field(),
            )
        });

        let guards = product.guards();
        let mut forms = FactorForms::new(widths.len());
        let answer = match &plan.evaluated {
            Some(blocks) => self.evaluate_product(&plan, blocks, &product, &mut forms),
            None => self.accumulate_columns(&plan, &product, &mut forms),
        };
        self.check_no_overflow(&plan, lhs, rhs, &mut forms, guards.check);
        answer
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
            let patterns: Vec<IntBits> = widths
                .iter()
                .enumerate()
                .map(|(index, width)| pattern.bit_range(index * self.limb_bits(), *width))
                .collect();
            return Factor {
                bounds: patterns.iter().map(BigUint::from).collect(),
                limbs: patterns.into_iter().map(Limb::Constant).collect(),
                witnessed: false,
            };
        }

        let (limbs, bounds) = self
            .chain_operand(value, bits, widths.len())
            .into_iter()
            .zip(full_bounds(widths))
            .zip(widths)
            .map(|((limb, full), width)| match self.known(limb) {
                Some(pattern) => {
                    let pattern = pattern.cast(*width);
                    let bound = BigUint::from(&pattern);
                    (Limb::Constant(pattern), bound)
                }
                None => (Limb::Value(limb), full),
            })
            .unzip();
        Factor {
            limbs,
            witnessed: self.is_witness(value),
            bounds,
        }
    }

    fn evaluate_product(
        &mut self,
        plan: &ProductPlan,
        blocks: &[Block],
        product: &Schoolbook<'_>,
        forms: &mut FactorForms,
    ) -> Vec<ValueId> {
        let Schoolbook {
            lhs,
            rhs,
            addend,
            pinned,
            widths,
            ..
        } = *product;
        let num_limbs = widths.len();
        let hint = CastTarget::Int(plan.hint_bits);
        let guards = product.guards();
        let mask = guards.check.map(|condition| {
            let field = self.cast(condition, CastTarget::Field);
            let pure = self.pure_of(condition, self.is_witness(condition));
            (field, self.cast(pure, hint.clone()))
        });

        let lhs_fields: Vec<ValueId> = (0..num_limbs)
            .map(|index| {
                let limb = forms.field(self, Side::Lhs, index, lhs);
                match mask {
                    Some((mask_felt, _)) => self.bin(BinaryArithOpKind::UMul, mask_felt, limb),
                    None => limb,
                }
            })
            .collect();
        let rhs_fields: Vec<ValueId> = (0..num_limbs)
            .map(|index| forms.field(self, Side::Rhs, index, rhs))
            .collect();
        let target = pinned.map(|target| match guards.pin {
            Some(mut condition) => {
                condition = self.cast(condition, CastTarget::Field);
                target
                    .iter()
                    .map(|target_limb| self.bin(BinaryArithOpKind::UMul, condition, *target_limb))
                    .collect()
            }
            None => target.to_vec(),
        });

        let mut answer = Vec::with_capacity(num_limbs);
        let mut carries = Vec::new();
        let mut carry_in: Option<ValueId> = None;
        for (column_idx, &width) in widths.iter().enumerate() {
            let mut running = None;
            for left in 0..=column_idx {
                let a = forms.pure(self, Side::Lhs, left, lhs, &hint);
                let b = forms.pure(self, Side::Rhs, column_idx - left, rhs, &hint);
                let product = self.bin(BinaryArithOpKind::UMul, a, b);
                running = Some(match running {
                    None => product,
                    Some(sum) => self.bin(BinaryArithOpKind::UAdd, sum, product),
                });
            }
            let mut running = running.expect("a column has at least one partial product");
            if let Some((_, flag)) = mask {
                running = self.bin(BinaryArithOpKind::UMul, flag, running);
            }
            if let Some(addend) = addend {
                let addend_limb = forms.pure(self, Side::Addend, column_idx, addend, &hint);
                running = self.bin(BinaryArithOpKind::UAdd, running, addend_limb);
            }
            if let Some(carry) = carry_in {
                running = self.bin(BinaryArithOpKind::UAdd, running, carry);
            }

            answer.push(match &target {
                Some(target) => target[column_idx],
                None => {
                    let low = self.cast(running, CastTarget::Int(width));
                    let low = self.cast(low, CastTarget::Field);
                    let written = self.write_witness(low);
                    self.push_guarded(
                        guards.check,
                        OpCode::Rangecheck {
                            value: written,
                            max_bits: width,
                        },
                    );
                    written
                }
            });
            if column_idx + 1 < num_limbs {
                let carry = self.shifted_down(running, plan.hint_bits, width);
                let carry_field = self.cast(carry, CastTarget::Field);
                carries.push(self.write_witness(carry_field));
                carry_in = Some(carry);
            }
        }

        for block in blocks {
            if let Some(carry_bits) = block.carry_bits {
                self.push_guarded(
                    guards.check,
                    OpCode::Rangecheck {
                        value: carries[block.last],
                        max_bits: carry_bits,
                    },
                );
            }
        }

        let radix = self.field().two_pow(self.limb_bits());
        let remainder: Option<Vec<ValueId>> = addend.map(|addend| {
            (0..num_limbs)
                .map(|index| forms.field(self, Side::Addend, index, addend))
                .collect()
        });
        for point in 0..(2 * num_limbs - 1) as u64 {
            let a = self.evaluate_at(&lhs_fields, point);
            let b = self.evaluate_at(&rhs_fields, point);
            let mut c = self.evaluate_at(&answer, point);
            if let Some(remainder) = &remainder {
                let r = self.evaluate_at(remainder, point);
                c = self.bin(BinaryArithOpKind::USub, c, r);
            }
            if !carries.is_empty() {
                let h = self.evaluate_at(&carries, point);
                let scale = self.field_const(radix - self.field().constant(point));
                let h = self.bin(BinaryArithOpKind::UMul, h, scale);
                c = self.bin(BinaryArithOpKind::UAdd, c, h);
            }
            self.push(OpCode::Constrain { a, b, c });
        }
        answer
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

    /// Every column of a summed product's answer, as field elements, following `plan`.
    ///
    /// Where `product` is pinned, a column's final value is held equal to the target's limb, which
    /// bounds it as tightly as the range check it replaces; a column that never needed reducing is
    /// held to that limb too. The range checks are under `checks` and the equalities under `pins`.
    fn accumulate_columns(
        &mut self,
        plan: &ProductPlan,
        product: &Schoolbook<'_>,
        forms: &mut FactorForms,
    ) -> Vec<ValueId> {
        let Schoolbook {
            lhs,
            rhs,
            addend,
            pinned,
            widths,
            ..
        } = *product;
        let guards = product.guards();
        let hint = CastTarget::Int(plan.hint_bits);
        let mut answer = Vec::with_capacity(widths.len());
        let mut incoming: std::collections::VecDeque<(ValueId, ValueId)> = Default::default();

        for (index, (steps, &width)) in plan.columns.iter().zip(widths).enumerate() {
            let mut outgoing = Vec::new();
            let mut column = Accumulator { sum: None };
            for (position, step) in steps.iter().enumerate() {
                // The one check a pinned column's equality takes the place of.
                let bounds_the_answer = pinned.is_none() || position + 1 < steps.len();
                match *step {
                    Step::Product {
                        lhs: left,
                        rhs: right,
                    } => {
                        let a = forms.field(self, Side::Lhs, left, lhs);
                        let b = forms.field(self, Side::Rhs, right, rhs);
                        let product = self.bin(BinaryArithOpKind::UMul, a, b);
                        let a = forms.pure(self, Side::Lhs, left, lhs, &hint);
                        let b = forms.pure(self, Side::Rhs, right, rhs, &hint);
                        let hinted = self.bin(BinaryArithOpKind::UMul, a, b);
                        self.accumulate(&mut column, product, hinted);
                    }
                    Step::Addend(limb) => {
                        let addend = addend.expect("the plan adds limbs of an addend it was given");
                        let value = forms.field(self, Side::Addend, limb, addend);
                        let hinted = forms.pure(self, Side::Addend, limb, addend, &hint);
                        self.accumulate(&mut column, value, hinted);
                    }
                    Step::Carry => {
                        let (carry, hinted) = incoming
                            .pop_front()
                            .expect("the plan adds each carry the column below made");
                        self.accumulate(&mut column, carry, hinted);
                    }
                    Step::Reduce { carry_bits } => {
                        let (sum, hinted) = column
                            .sum
                            .expect("the plan reduces only a column that has a bound");
                        match carry_bits {
                            None => {
                                if bounds_the_answer {
                                    self.push_guarded(
                                        guards.check,
                                        OpCode::Rangecheck {
                                            value: sum,
                                            max_bits: width,
                                        },
                                    );
                                }
                            }
                            Some(carry_bits) => {
                                let carry = self.shifted_down(hinted, plan.hint_bits, width);
                                let carry = self.cast(carry, CastTarget::Int(carry_bits));
                                let carry_field = self.cast(carry, CastTarget::Field);
                                let written = self.write_witness(carry_field);
                                self.push_guarded(
                                    guards.check,
                                    OpCode::Rangecheck {
                                        value: written,
                                        max_bits: carry_bits,
                                    },
                                );
                                let place = self.field_const(self.field().two_pow(width));
                                let scaled = self.bin(BinaryArithOpKind::UMul, written, place);
                                let low = self.bin(BinaryArithOpKind::USub, sum, scaled);
                                if bounds_the_answer {
                                    self.push_guarded(
                                        guards.check,
                                        OpCode::Rangecheck {
                                            value: low,
                                            max_bits: width,
                                        },
                                    );
                                }

                                let low_hint = self.cast(hinted, CastTarget::Int(width));
                                let low_hint = self.cast(low_hint, hint.clone());
                                column.sum = Some((low, low_hint));
                                outgoing.push((written, self.cast(carry, hint.clone())));
                            }
                        }
                    }
                }
            }
            let value = match column.sum {
                Some((sum, _)) => sum,
                None => self.field_const(self.field().zero()),
            };
            answer.push(match pinned {
                Some(target) => {
                    self.constrain_equal(value, target[index], guards.pin);
                    target[index]
                }
                None => value,
            });
            incoming.extend(outgoing);
        }
        assert!(
            incoming.is_empty(),
            "ICE: the top column of a product carried out of the answer"
        );
        answer
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

    fn accumulate(&mut self, column: &mut Accumulator, term: ValueId, hinted: ValueId) {
        column.sum = Some(match column.sum {
            None => (term, hinted),
            Some((sum, sum_hint)) => (
                self.bin(BinaryArithOpKind::UAdd, sum, term),
                self.bin(BinaryArithOpKind::UAdd, sum_hint, hinted),
            ),
        });
    }

    /// The partial products at or past the top of the answer, held to zero one left limb at a time.
    fn check_no_overflow(
        &mut self,
        plan: &ProductPlan,
        lhs: &Factor,
        rhs: &Factor,
        forms: &mut FactorForms,
        guard: Option<ValueId>,
    ) {
        let zero = self.field_const(self.field().zero());
        for (left, group) in &plan.overflow {
            let a = forms.field(self, Side::Lhs, *left, lhs);
            let mut sum = None;
            for right in group {
                let b = forms.field(self, Side::Rhs, *right, rhs);
                sum = Some(match sum {
                    None => b,
                    Some(sum) => self.bin(BinaryArithOpKind::UAdd, sum, b),
                });
            }
            let sum = sum.expect("an overflow group names at least one limb");
            let (a, b) = match guard {
                None => (a, sum),
                Some(condition) => {
                    let flag = self.cast(condition, CastTarget::Field);
                    (flag, self.bin(BinaryArithOpKind::UMul, a, sum))
                }
            };
            self.push(OpCode::Constrain { a, b, c: zero });
        }
    }
}

/// Which operand of a product a limb belongs to.
#[derive(Clone, Copy)]
enum Side {
    Lhs,
    Rhs,
    Addend,
}

/// The field element and the widened pure hint of each operand limb.
struct FactorForms {
    field: [Vec<Option<ValueId>>; 3],
    pure: [Vec<Option<ValueId>>; 3],
}

impl FactorForms {
    fn new(count: usize) -> Self {
        Self {
            field: std::array::from_fn(|_| vec![None; count]),
            pure: std::array::from_fn(|_| vec![None; count]),
        }
    }

    fn field(
        &mut self,
        rewriter: &mut Rewriter<'_>,
        side: Side,
        index: usize,
        factor: &Factor,
    ) -> ValueId {
        *self.field[side as usize][index]
            .get_or_insert_with(|| rewriter.limb_field(&factor.limbs[index]))
    }

    fn pure(
        &mut self,
        rewriter: &mut Rewriter<'_>,
        side: Side,
        index: usize,
        factor: &Factor,
        hint: &CastTarget,
    ) -> ValueId {
        *self.pure[side as usize][index].get_or_insert_with(|| match &factor.limbs[index] {
            Limb::Value(limb) => {
                let pure = rewriter.pure_of(*limb, factor.witnessed);
                rewriter.cast(pure, hint.clone())
            }
            Limb::Constant(pattern) => {
                let CastTarget::Int(bits) = hint else {
                    ice_unreachable!("a hint is an integer width");
                };
                rewriter.int_const(pattern.cast(*bits))
            }
        })
    }
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
