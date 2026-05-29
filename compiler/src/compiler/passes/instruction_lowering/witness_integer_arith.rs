use ark_ff::{AdditiveGroup as _, Field as _, PrimeField};
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, Zero};

use crate::compiler::{
    Field,
    analysis::{types::FunctionTypeInfo, value_range_analysis::IntInterval},
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessIntegerArithOps {}

impl InstructionLoweringRule for LowerWitnessIntegerArithOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_arith(b, context, Some(*condition), inner.as_ref())
        } else {
            self.process_arith(b, context, None, instruction)
        }
    }
}

impl LowerWitnessIntegerArithOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_arith(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Add | BinaryArithOpKind::Sub),
                result,
                lhs,
                rhs,
            } if self.should_lower_integer_arith(context, *lhs, *rhs) => {
                let (bits, signed) =
                    integer_bits_and_signedness(context.types().get_value_type(*lhs)).unwrap();
                if signed {
                    self.lower_signed_addsub(b, context, guard, *kind, *result, *lhs, *rhs, bits);
                } else {
                    self.lower_unsigned_addsub(b, context, guard, *kind, *result, *lhs, *rhs, bits);
                }
                true
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result,
                lhs,
                rhs,
            } if self.should_lower_integer_arith(context, *lhs, *rhs) => {
                let (bits, signed) =
                    integer_bits_and_signedness(context.types().get_value_type(*lhs)).unwrap();
                if signed {
                    self.lower_signed_mul(b, context, guard, *result, *lhs, *rhs, bits);
                } else {
                    self.lower_unsigned_mul(b, context, guard, *result, *lhs, *rhs, bits);
                }
                true
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result,
                lhs,
                rhs,
            } if self.should_lower_integer_arith(context, *lhs, *rhs) => {
                let (bits, signed) =
                    integer_bits_and_signedness(context.types().get_value_type(*lhs)).unwrap();
                if signed {
                    self.lower_signed_divmod(b, context, guard, *kind, *result, *lhs, *rhs, bits);
                } else {
                    self.lower_unsigned_divmod_result(
                        b, context, guard, *kind, *result, *lhs, *rhs, bits,
                    );
                }
                true
            }
            _ => false,
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
        let rc_bits = narrow_rangecheck_width(&range, bits);
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
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let product_range = context.range(lhs).mul(&context.range(rhs));
        assert!(
            range_fits_field_injectively(&product_range),
            "unsigned multiplication product range is too wide for a single-field product"
        );

        let value = b.mul(lhs_field, rhs_field);
        let rc_bits = narrow_rangecheck_width(&product_range, bits);
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
        let lhs_range = context.range(lhs);
        let rhs_range = context.range(rhs);
        let sign_l = if lhs_range.is_non_negative_in_signed(bits) {
            b.field_const(Field::ZERO)
        } else {
            let sign_l_bits = b.bit_range(lhs, bits - 1, 1);
            b.cast_to_field(sign_l_bits)
        };
        let sign_r = if rhs_range.is_non_negative_in_signed(bits) {
            b.field_const(Field::ZERO)
        } else {
            let sign_r_bits = b.bit_range(rhs, bits - 1, 1);
            b.cast_to_field(sign_r_bits)
        };
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let lhs_signed = signed_value_from_encoded(b, lhs_field, sign_l, bits);
        let rhs_signed = signed_value_from_encoded(b, rhs_field, sign_r, bits);

        let result_range = match kind {
            BinaryArithOpKind::Add => lhs_range.add(&rhs_range),
            BinaryArithOpKind::Sub => lhs_range.sub(&rhs_range),
            _ => unreachable!(),
        };
        assert!(
            range_fits_field_injectively(&result_range),
            "signed add/sub result range is too wide for a single-field equality"
        );

        let signed_raw = match kind {
            BinaryArithOpKind::Add => b.add(lhs_signed, rhs_signed),
            BinaryArithOpKind::Sub => b.sub(lhs_signed, rhs_signed),
            _ => unreachable!(),
        };

        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let result_hint = match kind {
            BinaryArithOpKind::Add => b.add(lhs_pure, rhs_pure),
            BinaryArithOpKind::Sub => b.sub(lhs_pure, rhs_pure),
            _ => unreachable!(),
        };
        let result_hint_unsigned = b.cast_to(CastTarget::U(bits), result_hint);
        let result_hint_field = b.cast_to_field(result_hint_unsigned);
        let result_wit = b.write_witness(result_hint_field);
        guarded_rangecheck(b, result_wit, bits, guard);
        let sign_result = if result_range.is_non_negative_in_signed(bits) {
            b.field_const(Field::ZERO)
        } else {
            let result_int = b.cast_to(CastTarget::U(bits), result_wit);
            let sign_result_bits = b.bit_range(result_int, bits - 1, 1);
            b.cast_to_field(sign_result_bits)
        };
        let result_signed = signed_value_from_encoded(b, result_wit, sign_result, bits);

        let diff = b.sub(signed_raw, result_signed);
        let zero = b.field_const(Field::ZERO);
        let flag = one_or_condition_field(b, context.types(), guard);
        b.constrain(flag, diff, zero);

        let result_value = guarded_or_zero_field(b, result_wit, guard);
        b.emit(OpCode::Cast {
            result,
            value: result_value,
            target: CastTarget::I(bits),
        });
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
        let lhs_range = context.range(lhs);
        let rhs_range = context.range(rhs);
        let product_range = lhs_range.mul(&rhs_range);
        assert!(
            range_fits_field_injectively(&product_range),
            "signed multiplication product range is too wide for a single-field product"
        );

        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let sign_l = if lhs_range.is_non_negative_in_signed(bits) {
            b.field_const(Field::ZERO)
        } else {
            let sign_l_bits = b.bit_range(lhs, bits - 1, 1);
            b.cast_to_field(sign_l_bits)
        };
        let sign_r = if rhs_range.is_non_negative_in_signed(bits) {
            b.field_const(Field::ZERO)
        } else {
            let sign_r_bits = b.bit_range(rhs, bits - 1, 1);
            b.cast_to_field(sign_r_bits)
        };
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let lhs_signed = signed_value_from_encoded(b, lhs_field, sign_l, bits);
        let rhs_signed = signed_value_from_encoded(b, rhs_field, sign_r, bits);

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let result_hint = b.mul(lhs_pure, rhs_pure);
        let result_hint_unsigned = b.cast_to(CastTarget::U(bits), result_hint);
        let result_hint_field = b.cast_to_field(result_hint_unsigned);
        let result_wit = b.write_witness(result_hint_field);
        guarded_rangecheck(b, result_wit, bits, guard);
        let sign_result = if product_range.is_non_negative_in_signed(bits) {
            b.field_const(Field::ZERO)
        } else {
            let result_int = b.cast_to(CastTarget::U(bits), result_wit);
            let sign_result_bits = b.bit_range(result_int, bits - 1, 1);
            b.cast_to_field(sign_result_bits)
        };
        let result_signed = signed_value_from_encoded(b, result_wit, sign_result, bits);

        let product = b.mul(lhs_signed, rhs_signed);
        let diff = b.sub(product, result_signed);
        let zero = b.field_const(Field::ZERO);
        let flag = one_or_condition_field(b, context.types(), guard);
        b.constrain(flag, diff, zero);

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
        let lhs_range = context.range(lhs);
        let rhs_range = context.range(rhs);

        let sign_l_is_witness = lhs_witness && !lhs_range.is_non_negative_in_signed(bits);
        let (sign_l_u1, sign_l) = if lhs_range.is_non_negative_in_signed(bits) {
            (b.u_const(1, 0), b.field_const(Field::ZERO))
        } else {
            let sign_l_bits = b.bit_range(lhs, bits - 1, 1);
            let sign_l_u1 = b.cast_to(CastTarget::U(1), sign_l_bits);
            let sign_l = b.cast_to_field(sign_l_u1);
            (sign_l_u1, sign_l)
        };
        let sign_r_is_witness = if lhs == rhs {
            sign_l_is_witness
        } else {
            rhs_witness && !rhs_range.is_non_negative_in_signed(bits)
        };
        let (sign_r_u1, sign_r) = if lhs == rhs {
            (sign_l_u1, sign_l)
        } else if rhs_range.is_non_negative_in_signed(bits) {
            (b.u_const(1, 0), b.field_const(Field::ZERO))
        } else {
            let sign_r_bits = b.bit_range(rhs, bits - 1, 1);
            let sign_r_u1 = b.cast_to(CastTarget::U(1), sign_r_bits);
            let sign_r = b.cast_to_field(sign_r_u1);
            (sign_r_u1, sign_r)
        };
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);

        let abs_l = self.write_abs_value(
            b,
            lhs,
            lhs_field,
            sign_l,
            sign_l_u1,
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
                sign_r_u1,
                bits,
                rhs_witness,
                sign_r_is_witness,
                guard,
            )
        };

        let abs_l_range = abs_bound(&lhs_range);
        let abs_r_range = if lhs == rhs {
            abs_l_range.clone()
        } else {
            abs_bound(&rhs_range)
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

        let quotient_sign_u1 = b.xor(sign_l_u1, sign_r_u1);
        let quotient_sign = b.cast_to_field(quotient_sign_u1);
        let quotient_sign_is_witness = sign_l_is_witness || sign_r_is_witness;
        if quotient_sign_is_witness {
            guarded_rangecheck(b, quotient_sign, 1, guard);
        }

        let quotient = self.write_signed_magnitude_result(
            b,
            divmod.q,
            quotient_sign,
            quotient_sign_u1,
            quotient_sign_is_witness,
            bits,
            guard,
        );
        let remainder = self.write_signed_magnitude_result(
            b,
            divmod.r,
            sign_l,
            sign_l_u1,
            sign_l_is_witness,
            bits,
            guard,
        );

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
        sign_u1: ValueId,
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
            b.value_of(sign_u1)
        } else {
            sign_u1
        };
        let abs_hint = b.select(sign_for_hint, neg, pure_value);
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
        sign_u1: ValueId,
        sign_is_witness: bool,
        bits: usize,
        guard: Option<ValueId>,
    ) -> ValueId {
        let magnitude_pure = b.value_of(magnitude);
        let sign_for_hint = if sign_is_witness {
            b.value_of(sign_u1)
        } else {
            sign_u1
        };
        let magnitude_field = b.cast_to_field(magnitude_pure);
        let two_n_field = b.field_const(two_pow(bits));
        let neg = b.sub(two_n_field, magnitude_field);
        let encoded_if_nonzero = b.select(sign_for_hint, neg, magnitude_field);
        let zero = b.field_const(Field::ZERO);
        let magnitude_is_zero = b.eq(magnitude_field, zero);
        let encoded_hint = b.select(magnitude_is_zero, zero, encoded_if_nonzero);
        let encoded = b.write_witness(encoded_hint);
        guarded_rangecheck(b, encoded, bits, guard);

        let encoded_int = b.cast_to(CastTarget::U(bits), encoded);
        let result_sign_bits = b.bit_range(encoded_int, bits - 1, 1);
        let result_sign = b.cast_to_field(result_sign_bits);
        let signed_result = signed_value_from_encoded(b, encoded, result_sign, bits);

        let two = b.field_const(Field::from(2));
        let two_sign = b.mul(two, sign);
        let one = b.field_const(Field::ONE);
        let factor = b.sub(one, two_sign);
        b.constrain(magnitude, factor, signed_result);
        encoded
    }
}

fn two_pow(exponent: usize) -> Field {
    Field::from(2).pow([exponent as u64])
}

fn bn254_modulus() -> BigInt {
    let limbs = <Field as PrimeField>::MODULUS.0;
    let bytes_le: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigInt::from_bytes_le(Sign::Plus, &bytes_le)
}

fn condition_field(
    b: &mut HLBlockEmitter<'_>,
    types: &FunctionTypeInfo,
    condition: ValueId,
) -> ValueId {
    if types.get_value_type(condition).strip_witness().is_field() {
        condition
    } else {
        b.cast_to_field(condition)
    }
}

fn one_or_condition_field(
    b: &mut HLBlockEmitter<'_>,
    types: &FunctionTypeInfo,
    guard: Option<ValueId>,
) -> ValueId {
    guard
        .map(|condition| condition_field(b, types, condition))
        .unwrap_or_else(|| b.field_const(Field::ONE))
}

fn guarded_rangecheck(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    bits: usize,
    guard: Option<ValueId>,
) {
    assert!(bits >= 1, "rangecheck width must be at least 1 bit");
    let rangecheck = OpCode::Rangecheck {
        value,
        max_bits: bits,
    };
    if let Some(condition) = guard {
        b.emit(OpCode::Guard {
            condition,
            inner: Box::new(rangecheck),
        });
    } else {
        b.emit(rangecheck);
    }
}

fn guarded_or_zero_field(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    guard: Option<ValueId>,
) -> ValueId {
    if let Some(condition) = guard {
        let zero = b.field_const(Field::ZERO);
        b.select(condition, value, zero)
    } else {
        value
    }
}

fn integer_bits_and_signedness(ty: &Type) -> Option<(usize, bool)> {
    match ty.strip_witness().expr {
        TypeExpr::U(bits) => Some((bits, false)),
        TypeExpr::I(bits) => Some((bits, true)),
        _ => None,
    }
}

fn unsigned_bit_width(range: &IntInterval) -> Option<usize> {
    let lo = range.lo()?;
    let hi = range.hi()?;
    if lo.is_negative() {
        return None;
    }
    Some(hi.bits() as usize)
}

fn narrow_rangecheck_width(range: &IntInterval, default_bits: usize) -> usize {
    let Some(width) = unsigned_bit_width(range) else {
        return default_bits;
    };
    width.max(1).min(default_bits)
}

fn range_fits_field_injectively(range: &IntInterval) -> bool {
    let Some(lo) = range.lo() else {
        return false;
    };
    let Some(hi) = range.hi() else {
        return false;
    };
    let p = bn254_modulus();
    // All integer representatives in this range have distinct BN254 field
    // encodings when their pairwise distance is less than p.
    hi - lo < p
}

fn signed_value_from_encoded(
    b: &mut HLBlockEmitter<'_>,
    encoded_field: ValueId,
    sign: ValueId,
    bits: usize,
) -> ValueId {
    let sign_shift = b.field_const(two_pow(bits));
    let sign_shifted = b.mul(sign, sign_shift);
    b.sub(encoded_field, sign_shifted)
}

struct DivModResult {
    q: ValueId,
    r: ValueId,
}

#[allow(clippy::too_many_arguments)]
fn lower_unsigned_divmod(
    b: &mut HLBlockEmitter<'_>,
    dividend: ValueId,
    divisor: ValueId,
    bits: usize,
    dividend_is_witness: bool,
    divisor_is_witness: bool,
    dividend_range: &IntInterval,
    divisor_range: &IntInterval,
    guard: Option<ValueId>,
    guard_is_witness: bool,
    guard_flag: ValueId,
) -> DivModResult {
    if dividend == divisor {
        let active = if let Some(condition) = guard {
            let condition = if guard_is_witness {
                b.value_of(condition)
            } else {
                condition
            };
            b.cast_to_field(condition)
        } else {
            b.field_const(Field::ONE)
        };
        let zero = b.field_const(Field::ZERO);
        let q_wit = b.write_witness(active);
        let r_wit = b.write_witness(zero);

        let one = b.field_const(Field::ONE);
        let q_diff = b.sub(q_wit, active);
        b.constrain(one, q_diff, zero);
        b.constrain(one, r_wit, zero);

        guarded_rangecheck(b, q_wit, 1, guard);
        guarded_rangecheck(b, r_wit, 1, guard);

        let divisor_field = b.cast_to_field(divisor);
        let divisor_minus_one = b.sub(divisor_field, one);
        guarded_rangecheck(b, divisor_minus_one, bits, guard);

        return DivModResult { q: q_wit, r: r_wit };
    }

    let dividend_pure = if dividend_is_witness {
        b.value_of(dividend)
    } else {
        dividend
    };
    let divisor_pure = if divisor_is_witness {
        b.value_of(divisor)
    } else {
        divisor
    };

    let mut dividend_hint = b.cast_to(CastTarget::U(bits), dividend_pure);
    let mut divisor_hint = b.cast_to(CastTarget::U(bits), divisor_pure);
    if let Some(condition) = guard {
        let condition = if guard_is_witness {
            b.value_of(condition)
        } else {
            condition
        };
        let zero = b.u_const(bits, 0);
        let one = b.u_const(bits, 1);
        dividend_hint = b.select(condition, dividend_hint, zero);
        divisor_hint = b.select(condition, divisor_hint, one);
    }
    let q_hint = b.div(dividend_hint, divisor_hint);
    let qb = b.mul(q_hint, divisor_hint);
    let r_hint = b.sub(dividend_hint, qb);
    let q_hint_field = b.cast_to_field(q_hint);
    let r_hint_field = b.cast_to_field(r_hint);
    let q_wit = b.write_witness(q_hint_field);
    let r_wit = b.write_witness(r_hint_field);

    let dividend_field = b.cast_to_field(dividend);
    let divisor_field = b.cast_to_field(divisor);
    let dividend_minus_r = b.sub(dividend_field, r_wit);
    if guard.is_some() {
        let product = b.mul(q_wit, divisor_field);
        let diff = b.sub(product, dividend_minus_r);
        let zero = b.field_const(Field::ZERO);
        b.constrain(guard_flag, diff, zero);
    } else {
        b.constrain(q_wit, divisor_field, dividend_minus_r);
    }

    let q_bits = narrow_rangecheck_width(&quotient_bound(dividend_range, divisor_range), bits);
    let r_bound = remainder_bound(divisor_range);
    let r_bits = narrow_rangecheck_width(&r_bound, bits);
    guarded_rangecheck(b, q_wit, q_bits, guard);
    guarded_rangecheck(b, r_wit, r_bits, guard);

    let one = b.field_const(Field::ONE);
    let divisor_minus_r = b.sub(divisor_field, r_wit);
    let divisor_minus_r_minus_one = b.sub(divisor_minus_r, one);
    guarded_rangecheck(b, divisor_minus_r_minus_one, r_bits, guard);

    DivModResult { q: q_wit, r: r_wit }
}

fn quotient_bound(a_range: &IntInterval, b_range: &IntInterval) -> IntInterval {
    let (Some(a_hi), Some(b_lo)) = (a_range.hi(), b_range.lo()) else {
        return IntInterval::top();
    };
    if !a_range.is_non_negative() || !b_lo.is_positive() {
        return IntInterval::top();
    }
    IntInterval::closed(BigInt::zero(), a_hi / b_lo)
}

fn remainder_bound(b_range: &IntInterval) -> IntInterval {
    let Some(b_hi) = b_range.hi() else {
        return IntInterval::top();
    };
    if !b_hi.is_positive() {
        return IntInterval::top();
    }
    IntInterval::closed(BigInt::zero(), b_hi - BigInt::one())
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
