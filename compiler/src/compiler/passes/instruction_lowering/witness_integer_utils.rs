use ark_ff::{AdditiveGroup as _, Field as _, PrimeField};
use num_bigint::{BigInt, Sign};
use num_traits::Signed;

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
