use ark_ff::{AdditiveGroup as _, Field as _, PrimeField};
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, Zero};

use crate::compiler::{
    Field,
    analysis::{types::FunctionTypeInfo, value_range_analysis::IntInterval},
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

pub(crate) fn two_pow(exponent: usize) -> Field {
    Field::from(2).pow([exponent as u64])
}

pub(crate) fn bn254_modulus() -> BigInt {
    let limbs = <Field as PrimeField>::MODULUS.0;
    let bytes_le: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigInt::from_bytes_le(Sign::Plus, &bytes_le)
}

pub(crate) fn condition_field(
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

pub(crate) fn one_or_condition_field(
    b: &mut HLBlockEmitter<'_>,
    types: &FunctionTypeInfo,
    guard: Option<ValueId>,
) -> ValueId {
    guard
        .map(|condition| condition_field(b, types, condition))
        .unwrap_or_else(|| b.field_const(Field::ONE))
}

pub(crate) fn guarded_rangecheck(
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

pub(crate) fn guarded_or_zero_field(
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

pub(crate) fn cast_target_for_integer_type(ty: &Type) -> CastTarget {
    match ty.strip_witness().expr {
        TypeExpr::U(bits) => CastTarget::U(bits),
        TypeExpr::I(bits) => CastTarget::I(bits),
        other => panic!("expected integer type, got {:?}", other),
    }
}

pub(crate) fn integer_bits_and_signedness(ty: &Type) -> Option<(usize, bool)> {
    match ty.strip_witness().expr {
        TypeExpr::U(bits) => Some((bits, false)),
        TypeExpr::I(bits) => Some((bits, true)),
        _ => None,
    }
}

pub(crate) fn unsigned_bit_width(range: &IntInterval) -> Option<usize> {
    let lo = range.lo()?;
    let hi = range.hi()?;
    if lo.is_negative() {
        return None;
    }
    Some(hi.bits() as usize)
}

pub(crate) fn narrow_rangecheck_width(range: &IntInterval, default_bits: usize) -> usize {
    let Some(width) = unsigned_bit_width(range) else {
        return default_bits;
    };
    width.max(1).min(default_bits)
}

pub(crate) fn quotient_bound(a_range: &IntInterval, b_range: &IntInterval) -> IntInterval {
    let (Some(a_hi), Some(b_lo)) = (a_range.hi(), b_range.lo()) else {
        return IntInterval::top();
    };
    if !a_range.is_non_negative() || !b_lo.is_positive() {
        return IntInterval::top();
    }
    IntInterval::closed(BigInt::zero(), a_hi / b_lo)
}

pub(crate) fn remainder_bound(b_range: &IntInterval) -> IntInterval {
    let Some(b_hi) = b_range.hi() else {
        return IntInterval::top();
    };
    if !b_hi.is_positive() {
        return IntInterval::top();
    }
    IntInterval::closed(BigInt::zero(), b_hi - BigInt::one())
}

pub(crate) fn range_fits_field_injectively(range: &IntInterval) -> bool {
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

pub(crate) fn emit_bit_range(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    offset: usize,
    width: usize,
    source_width: Option<usize>,
) -> ValueId {
    let result = b.fresh_value();
    b.emit(OpCode::BitRange {
        result,
        value,
        offset,
        width,
        source_width,
    });
    result
}

#[derive(Clone, Copy)]
pub(crate) enum SignBitSource {
    Integer,
    Field,
}

pub(crate) fn extract_sign_bit(
    b: &mut HLBlockEmitter<'_>,
    encoded: ValueId,
    bits: usize,
    value_range: &IntInterval,
    source: SignBitSource,
) -> ValueId {
    if value_range.is_non_negative_in_signed(bits) {
        return b.field_const(Field::ZERO);
    }
    assert!(bits >= 1, "signed integer width must be at least 1 bit");
    let source_width = match source {
        SignBitSource::Integer => None,
        SignBitSource::Field => Some(bits),
    };
    let sign = emit_bit_range(b, encoded, bits - 1, 1, source_width);
    match source {
        SignBitSource::Integer => b.cast_to_field(sign),
        SignBitSource::Field => sign,
    }
}

pub(crate) fn signed_value_from_encoded(
    b: &mut HLBlockEmitter<'_>,
    encoded_field: ValueId,
    sign: ValueId,
    bits: usize,
) -> ValueId {
    let sign_shift = b.field_const(two_pow(bits));
    let sign_shifted = b.mul(sign, sign_shift);
    b.sub(encoded_field, sign_shifted)
}

pub(crate) fn xor_bits(
    b: &mut HLBlockEmitter<'_>,
    lhs: ValueId,
    rhs: ValueId,
    lhs_is_witness: bool,
    rhs_is_witness: bool,
) -> ValueId {
    let product = if lhs_is_witness && rhs_is_witness {
        let lhs_pure = b.value_of(lhs);
        let rhs_pure = b.value_of(rhs);
        let product_hint = b.mul(lhs_pure, rhs_pure);
        let product = b.write_witness(product_hint);
        b.constrain(lhs, rhs, product);
        product
    } else {
        b.mul(lhs, rhs)
    };

    let two = b.field_const(Field::from(2));
    let two_product = b.mul(two, product);
    let sum = b.add(lhs, rhs);
    b.sub(sum, two_product)
}

pub(crate) struct DivModResult {
    pub q: ValueId,
    pub r: ValueId,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_unsigned_divmod(
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
        let product = if divisor_is_witness {
            let q_pure = b.value_of(q_wit);
            let divisor_pure = b.value_of(divisor_field);
            let product_hint = b.mul(q_pure, divisor_pure);
            let product = b.write_witness(product_hint);
            b.constrain(q_wit, divisor_field, product);
            product
        } else {
            b.mul(q_wit, divisor_field)
        };
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
