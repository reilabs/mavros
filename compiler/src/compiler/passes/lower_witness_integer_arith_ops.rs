use ark_ff::{AdditiveGroup as _, Field as _};
use num_bigint::BigInt;
use num_traits::{Signed, Zero};

use crate::compiler::{
    Field,
    analysis::{flow_analysis::FlowAnalysis, value_range_analysis::IntInterval},
    pass_manager::AnalysisId,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, OpCode,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{
    lowering_pass::{LoweringContext, LoweringPass},
    witness_integer_utils::{
        SignBitSource, extract_sign_bit, guarded_or_zero_field, guarded_rangecheck,
        integer_bits_and_signedness, lower_unsigned_divmod, one_or_condition_field,
        range_fits_field_injectively, signed_value_from_encoded, two_pow, xor_bits,
    },
};

pub struct LowerWitnessIntegerArithOps {}

impl LoweringPass for LowerWitnessIntegerArithOps {
    const NAME: &'static str = "lower_witness_integer_arith_ops";

    fn needs_value_ranges(&self) -> bool {
        true
    }

    fn preserved_analyses(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn process_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: OpCode,
    ) {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_arith(b, context, Some(condition), *inner);
        } else {
            self.process_arith(b, context, None, instruction);
        }
    }
}

impl LowerWitnessIntegerArithOps {
    pub fn new() -> Self {
        Self {}
    }

    fn emit_guarded(&self, b: &mut HLBlockEmitter<'_>, guard: Option<ValueId>, op: OpCode) {
        if let Some(condition) = guard {
            b.emit(OpCode::Guard {
                condition,
                inner: Box::new(op),
            });
        } else {
            b.emit(op);
        }
    }

    fn process_arith(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        op: OpCode,
    ) {
        match op {
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Add | BinaryArithOpKind::Sub),
                result,
                lhs,
                rhs,
            } if self.should_lower_integer_arith(context, lhs, rhs) => {
                let (bits, signed) =
                    integer_bits_and_signedness(context.types().get_value_type(lhs)).unwrap();
                if signed {
                    self.lower_signed_addsub(b, context, guard, kind, result, lhs, rhs, bits);
                } else {
                    self.lower_unsigned_addsub(b, context, guard, kind, result, lhs, rhs, bits);
                }
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result,
                lhs,
                rhs,
            } if self.should_lower_integer_arith(context, lhs, rhs) => {
                let (bits, signed) =
                    integer_bits_and_signedness(context.types().get_value_type(lhs)).unwrap();
                if signed {
                    self.lower_signed_mul(b, context, guard, result, lhs, rhs, bits);
                } else {
                    self.lower_unsigned_mul(b, context, guard, result, lhs, rhs, bits);
                }
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result,
                lhs,
                rhs,
            } if self.should_lower_integer_arith(context, lhs, rhs) => {
                let (bits, signed) =
                    integer_bits_and_signedness(context.types().get_value_type(lhs)).unwrap();
                if signed {
                    self.lower_signed_divmod(b, context, guard, kind, result, lhs, rhs, bits);
                } else {
                    self.lower_unsigned_divmod_result(
                        b, context, guard, kind, result, lhs, rhs, bits,
                    );
                }
            }
            other => self.emit_guarded(b, guard, other),
        }
    }

    fn should_lower_integer_arith(
        &self,
        context: &LoweringContext<'_>,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let lhs_ty = context.types().get_value_type(lhs);
        let rhs_ty = context.types().get_value_type(rhs);
        (lhs_ty.is_witness_of() || rhs_ty.is_witness_of())
            && integer_bits_and_signedness(lhs_ty).is_some()
    }

    fn lower_unsigned_addsub(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let value = match kind {
            BinaryArithOpKind::Add => b.add(lhs_field, rhs_field),
            BinaryArithOpKind::Sub => b.sub(lhs_field, rhs_field),
            _ => unreachable!(),
        };
        let range = match kind {
            BinaryArithOpKind::Add => context.range(lhs).add(&context.range(rhs)),
            BinaryArithOpKind::Sub => context.range(lhs).sub(&context.range(rhs)),
            _ => unreachable!(),
        };
        let rc_bits = super::witness_integer_utils::narrow_rangecheck_width(&range, bits);
        guarded_rangecheck(b, value, rc_bits, guard);
        let value = guarded_or_zero_field(b, value, guard);
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::U(bits),
        });
    }

    fn lower_unsigned_mul(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let product_range = context.range(lhs).mul(&context.range(rhs));
        assert!(
            range_fits_field_injectively(&product_range),
            "unsigned multiplication product range is too wide for a single-field product"
        );

        let value = if lhs_witness && rhs_witness {
            let lhs_pure = b.value_of(lhs_field);
            let rhs_pure = b.value_of(rhs_field);
            let hint = b.mul(lhs_pure, rhs_pure);
            let witness = b.write_witness(hint);
            b.constrain(lhs_field, rhs_field, witness);
            witness
        } else {
            b.mul(lhs_field, rhs_field)
        };
        let rc_bits = super::witness_integer_utils::narrow_rangecheck_width(&product_range, bits);
        guarded_rangecheck(b, value, rc_bits, guard);
        let value = guarded_or_zero_field(b, value, guard);
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::U(bits),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_signed_addsub(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let sign_l = extract_sign_bit(
            b,
            lhs,
            bits,
            &context.range(lhs),
            guard,
            SignBitSource::Integer,
        );
        let sign_r = extract_sign_bit(
            b,
            rhs,
            bits,
            &context.range(rhs),
            guard,
            SignBitSource::Integer,
        );

        let result_range = match kind {
            BinaryArithOpKind::Add => context.range(lhs).add(&context.range(rhs)),
            BinaryArithOpKind::Sub => context.range(lhs).sub(&context.range(rhs)),
            _ => unreachable!(),
        };

        let two_n = b.field_const(two_pow(bits));
        let zero = b.field_const(Field::ZERO);
        match kind {
            BinaryArithOpKind::Add => {
                let raw = b.add(lhs_field, rhs_field);
                let raw_for_hint = guarded_or_zero_field(b, raw, guard);
                let raw_pure = b.value_of(raw_for_hint);
                let hint_result = b.truncate(raw_pure, bits, 254);
                let hint_diff = b.sub(raw_pure, hint_result);
                let hint_carry = b.div(hint_diff, two_n);
                let carry = b.write_witness(hint_carry);
                let carry_shifted = b.mul(carry, two_n);
                let result_lc = b.sub(raw, carry_shifted);

                guarded_rangecheck(b, result_lc, bits, guard);
                guarded_rangecheck(b, carry, 1, guard);
                let sign_result = extract_sign_bit(
                    b,
                    result_lc,
                    bits,
                    &result_range,
                    guard,
                    SignBitSource::Field,
                );

                let lhs_overflow = b.add(carry, sign_result);
                let rhs_overflow = b.add(sign_l, sign_r);
                let diff = b.sub(lhs_overflow, rhs_overflow);
                let flag = one_or_condition_field(b, context.types(), guard);
                b.constrain(flag, diff, zero);

                let result_lc = guarded_or_zero_field(b, result_lc, guard);
                b.emit(OpCode::Cast {
                    result,
                    value: result_lc,
                    target: CastTarget::I(bits),
                });
            }
            BinaryArithOpKind::Sub => {
                let raw = b.sub(lhs_field, rhs_field);
                let raw_for_hint = guarded_or_zero_field(b, raw, guard);
                let raw_pure = b.value_of(raw_for_hint);
                let shifted = b.add(raw_pure, two_n);
                let hint_result = b.truncate(shifted, bits, 254);
                let hint_rem = b.sub(shifted, hint_result);
                let hint_borrow = b.div(hint_rem, two_n);
                let borrow = b.write_witness(hint_borrow);
                let lhs_full = b.add(raw, two_n);
                let borrow_shifted = b.mul(borrow, two_n);
                let result_lc = b.sub(lhs_full, borrow_shifted);

                guarded_rangecheck(b, result_lc, bits, guard);
                guarded_rangecheck(b, borrow, 1, guard);
                let sign_result = extract_sign_bit(
                    b,
                    result_lc,
                    bits,
                    &result_range,
                    guard,
                    SignBitSource::Field,
                );

                let one = b.field_const(Field::ONE);
                let lhs_overflow = b.add(borrow, sign_result);
                let lhs_overflow = b.add(lhs_overflow, sign_r);
                let rhs_overflow = b.add(one, sign_l);
                let diff = b.sub(lhs_overflow, rhs_overflow);
                let flag = one_or_condition_field(b, context.types(), guard);
                b.constrain(flag, diff, zero);

                let result_lc = guarded_or_zero_field(b, result_lc, guard);
                b.emit(OpCode::Cast {
                    result,
                    value: result_lc,
                    target: CastTarget::I(bits),
                });
            }
            _ => unreachable!(),
        }
    }

    fn lower_signed_mul(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let product_range = context.range(lhs).mul(&context.range(rhs));
        assert!(
            range_fits_field_injectively(&product_range),
            "signed multiplication product range is too wide for a single-field product"
        );

        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let sign_l = extract_sign_bit(
            b,
            lhs,
            bits,
            &context.range(lhs),
            guard,
            SignBitSource::Integer,
        );
        let sign_r = extract_sign_bit(
            b,
            rhs,
            bits,
            &context.range(rhs),
            guard,
            SignBitSource::Integer,
        );
        let lhs_signed = signed_value_from_encoded(b, lhs_field, sign_l, bits);
        let rhs_signed = signed_value_from_encoded(b, rhs_field, sign_r, bits);

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let result_hint = b.mul(lhs_pure, rhs_pure);
        let result_hint_unsigned = b.cast_to(CastTarget::U(bits), result_hint);
        let result_hint_field = b.cast_to_field(result_hint_unsigned);
        let result_wit = b.write_witness(result_hint_field);
        guarded_rangecheck(b, result_wit, bits, guard);
        let sign_result = extract_sign_bit(
            b,
            result_wit,
            bits,
            &product_range,
            guard,
            SignBitSource::Field,
        );
        let result_signed = signed_value_from_encoded(b, result_wit, sign_result, bits);

        let use_product_witness = lhs_witness && rhs_witness;
        if guard.is_some() || use_product_witness {
            let product = if use_product_witness {
                let lhs_signed_pure = b.value_of(lhs_signed);
                let rhs_signed_pure = b.value_of(rhs_signed);
                let product_hint = b.mul(lhs_signed_pure, rhs_signed_pure);
                let product_wit = b.write_witness(product_hint);
                b.constrain(lhs_signed, rhs_signed, product_wit);
                product_wit
            } else {
                b.mul(lhs_signed, rhs_signed)
            };
            let diff = b.sub(product, result_signed);
            let zero = b.field_const(Field::ZERO);
            let flag = one_or_condition_field(b, context.types(), guard);
            b.constrain(flag, diff, zero);
        } else {
            b.constrain(lhs_signed, rhs_signed, result_signed);
        }

        b.emit(OpCode::Cast {
            result,
            value: result_wit,
            target: CastTarget::I(bits),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_unsigned_divmod_result(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let guard_is_witness = guard
            .map(|condition| context.types().get_value_type(condition).is_witness_of())
            .unwrap_or(false);
        let guard_flag = one_or_condition_field(b, context.types(), guard);
        let divmod = lower_unsigned_divmod(
            b,
            lhs,
            rhs,
            bits,
            context.types().get_value_type(lhs).is_witness_of(),
            context.types().get_value_type(rhs).is_witness_of(),
            &context.range(lhs),
            &context.range(rhs),
            guard,
            guard_is_witness,
            guard_flag,
        );
        let value = match kind {
            BinaryArithOpKind::Div => divmod.q,
            BinaryArithOpKind::Mod => divmod.r,
            _ => unreachable!(),
        };
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::U(bits),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_signed_divmod(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let sign_l = extract_sign_bit(
            b,
            lhs,
            bits,
            &context.range(lhs),
            guard,
            SignBitSource::Integer,
        );
        let sign_r = if lhs == rhs {
            sign_l
        } else {
            extract_sign_bit(
                b,
                rhs,
                bits,
                &context.range(rhs),
                guard,
                SignBitSource::Integer,
            )
        };

        let sign_l_is_witness = lhs_witness && !context.range(lhs).is_non_negative_in_signed(bits);
        let sign_r_is_witness = if lhs == rhs {
            sign_l_is_witness
        } else {
            rhs_witness && !context.range(rhs).is_non_negative_in_signed(bits)
        };

        let abs_l = self.write_abs_value(
            b,
            lhs,
            lhs_field,
            sign_l,
            bits,
            lhs_witness,
            sign_l_is_witness,
            guard,
        );
        let abs_r = if lhs == rhs {
            abs_l
        } else {
            self.write_abs_value(
                b,
                rhs,
                rhs_field,
                sign_r,
                bits,
                rhs_witness,
                sign_r_is_witness,
                guard,
            )
        };

        let abs_l_range = abs_bound(&context.range(lhs));
        let abs_r_range = if lhs == rhs {
            abs_l_range.clone()
        } else {
            abs_bound(&context.range(rhs))
        };
        let guard_is_witness = guard
            .map(|condition| context.types().get_value_type(condition).is_witness_of())
            .unwrap_or(false);
        let guard_flag = one_or_condition_field(b, context.types(), guard);
        let divmod = lower_unsigned_divmod(
            b,
            abs_l,
            abs_r,
            bits,
            true,
            true,
            &abs_l_range,
            &abs_r_range,
            guard,
            guard_is_witness,
            guard_flag,
        );

        let quotient_sign = xor_bits(b, sign_l, sign_r, sign_l_is_witness, sign_r_is_witness);
        let quotient_sign_is_witness = sign_l_is_witness || sign_r_is_witness;
        if quotient_sign_is_witness {
            guarded_rangecheck(b, quotient_sign, 1, guard);
        }

        let quotient = self.write_signed_magnitude_result(
            b,
            divmod.q,
            quotient_sign,
            quotient_sign_is_witness,
            bits,
            guard,
        );
        let remainder =
            self.write_signed_magnitude_result(b, divmod.r, sign_l, sign_l_is_witness, bits, guard);

        let value = match kind {
            BinaryArithOpKind::Div => quotient,
            BinaryArithOpKind::Mod => remainder,
            _ => unreachable!(),
        };
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::I(bits),
        });
    }

    fn write_abs_value(
        &self,
        b: &mut HLBlockEmitter<'_>,
        value: ValueId,
        value_field: ValueId,
        sign: ValueId,
        bits: usize,
        value_is_witness: bool,
        sign_is_witness: bool,
        guard: Option<ValueId>,
    ) -> ValueId {
        let pure_value = if value_is_witness {
            b.value_of(value)
        } else {
            value
        };
        let zero = b.i_const(bits, 0);
        let neg = b.sub(zero, pure_value);
        let sign_for_hint = if sign_is_witness {
            b.value_of(sign)
        } else {
            sign
        };
        let sign_u1 = b.cast_to(CastTarget::U(1), sign_for_hint);
        let abs_hint = b.select(sign_u1, neg, pure_value);
        let abs_hint_field = b.cast_to_field(abs_hint);
        let abs_wit = b.write_witness(abs_hint_field);

        let signed_value = signed_value_from_encoded(b, value_field, sign, bits);
        let two = b.field_const(Field::from(2));
        let two_sign = b.mul(two, sign);
        let one = b.field_const(Field::ONE);
        let factor = b.sub(one, two_sign);
        b.constrain(signed_value, factor, abs_wit);
        guarded_rangecheck(b, abs_wit, bits, guard);
        abs_wit
    }

    fn write_signed_magnitude_result(
        &self,
        b: &mut HLBlockEmitter<'_>,
        magnitude: ValueId,
        sign: ValueId,
        sign_is_witness: bool,
        bits: usize,
        guard: Option<ValueId>,
    ) -> ValueId {
        let magnitude_pure = b.value_of(magnitude);
        let sign_for_hint = if sign_is_witness {
            b.value_of(sign)
        } else {
            sign
        };
        let sign_u1 = b.cast_to(CastTarget::U(1), sign_for_hint);
        let magnitude_field = b.cast_to_field(magnitude_pure);
        let two_n_field = b.field_const(two_pow(bits));
        let neg = b.sub(two_n_field, magnitude_field);
        let encoded_if_nonzero = b.select(sign_u1, neg, magnitude_field);
        let zero = b.field_const(Field::ZERO);
        let magnitude_is_zero = b.eq(magnitude_field, zero);
        let encoded_hint = b.select(magnitude_is_zero, zero, encoded_if_nonzero);
        let encoded = b.write_witness(encoded_hint);
        guarded_rangecheck(b, encoded, bits, guard);

        let sign_range = IntInterval::top();
        let result_sign =
            extract_sign_bit(b, encoded, bits, &sign_range, guard, SignBitSource::Field);
        let signed_result = signed_value_from_encoded(b, encoded, result_sign, bits);

        let two = b.field_const(Field::from(2));
        let two_sign = b.mul(two, sign);
        let one = b.field_const(Field::ONE);
        let factor = b.sub(one, two_sign);
        b.constrain(magnitude, factor, signed_result);
        encoded
    }
}

fn abs_bound(range: &IntInterval) -> IntInterval {
    let Some(lo) = range.lo() else {
        return IntInterval::top();
    };
    let Some(hi) = range.hi() else {
        return IntInterval::top();
    };
    let lo_abs = lo.abs();
    let hi_abs = hi.abs();
    let max = if lo_abs >= hi_abs {
        lo_abs.clone()
    } else {
        hi_abs.clone()
    };
    if lo <= &BigInt::zero() && hi >= &BigInt::zero() {
        IntInterval::closed(BigInt::zero(), max)
    } else {
        let min = if lo_abs <= hi_abs { lo_abs } else { hi_abs };
        IntInterval::closed(min, max)
    }
}
