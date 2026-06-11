//! Lowers canonical `BitRange` operations after the witness integer/bitwise passes have emitted
//! all bit selections.

use ark_ff::{AdditiveGroup as _, Field as _};

use crate::compiler::{
    Field,
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, Endianness, MAX_SUPPORTED_UNSIGNED_BITS, OpCode, Radix, Type,
            TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerBitRangeOps {}

impl InstructionLoweringRule for LowerBitRangeOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        match instruction {
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => {
                self.lower_bit_range(b, context, *result, *value, *offset, *width);
                true
            }
            _ => false,
        }
    }
}

impl LowerBitRangeOps {
    pub fn new() -> Self {
        Self {}
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    ) {
        assert!(width > 0, "BitRange width must be at least 1");
        let value_type = context.types().get_value_type(value);
        let source_bits = value_type.get_bit_size();
        assert!(
            offset + width <= source_bits,
            "BitRange({}, {}) exceeds source width {}",
            offset,
            width,
            source_bits
        );
        match (
            value_type.is_witness_of(),
            value_type.strip_witness().is_field(),
        ) {
            (true, true) => self.lower_witness_field_bit_range(b, result, value, offset, width),
            (true, false) => self.lower_witness_bit_range(b, context, result, value, offset, width),
            (false, _) => {
                self.lower_pure_bit_range(b, context.types(), result, value, offset, width)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_pure_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        types: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    ) {
        let value_type = types.get_value_type(value);
        let extracted = lower_pure_bit_range_value(b, value, value_type, offset, width);
        let target = cast_target_for_scalar_type(types.get_value_type(result));
        b.emit(OpCode::Cast {
            result,
            value: extracted,
            target,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_witness_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    ) {
        let value_type = context.types().get_value_type(value);
        let source_bits = value_type.get_bit_size();
        let pure_value = b.value_of(value);
        let hint =
            lower_pure_bit_range_value(b, pure_value, &value_type.strip_witness(), offset, width);
        let hint_field = b.cast_to_field(hint);

        let result_witness = b.write_witness(hint_field);
        b.emit(OpCode::Cast {
            result,
            value: result_witness,
            target: cast_target_for_scalar_type(context.types().get_value_type(result)),
        });

        let result_field = b.ensure_field(result, context.types().get_value_type(result));
        b.rangecheck(result_field, width);

        let low_bits = offset;
        let low = if low_bits == 0 {
            None
        } else {
            let low_hint =
                lower_pure_bit_range_value(b, pure_value, &value_type.strip_witness(), 0, low_bits);
            let low_hint = b.cast_to_field(low_hint);
            let low = b.write_witness(low_hint);
            b.rangecheck(low, low_bits);
            Some(low)
        };

        let high_bits = source_bits - offset - width;
        let high = if high_bits == 0 {
            None
        } else {
            let high_hint = lower_pure_bit_range_value(
                b,
                pure_value,
                &value_type.strip_witness(),
                offset + width,
                high_bits,
            );
            let high_hint = b.cast_to_field(high_hint);
            let high = b.write_witness(high_hint);
            b.rangecheck(high, high_bits);
            Some(high)
        };

        let mut reconstructed = low.unwrap_or_else(|| b.field_const(Field::ZERO));
        let result_shift = b.field_const(two_pow(offset));
        let result_shifted = b.mul(result_field, result_shift);
        reconstructed = b.add(reconstructed, result_shifted);
        if let Some(high) = high {
            let high_shift = b.field_const(two_pow(offset + width));
            let high_shifted = b.mul(high, high_shift);
            reconstructed = b.add(reconstructed, high_shifted);
        }

        let value_field = b.ensure_field(value, value_type);
        let diff = b.sub(value_field, reconstructed);
        let zero = b.field_const(Field::ZERO);
        let flag = b.field_const(Field::ONE);
        b.constrain(flag, diff, zero);
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_witness_field_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    ) {
        let flag = b.field_const(Field::ONE);
        let bytes = decompose_canonical_field_bytes(b, value, flag);
        let selected = lower_field_bit_range_from_bytes(b, &bytes, offset, width, flag);

        b.emit(OpCode::Cast {
            result,
            value: selected,
            target: Type::witness_of(Type::field()),
        });
    }
}

fn decompose_canonical_field_bytes(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    flag: ValueId,
) -> Vec<ValueId> {
    let modulus_hi = b.field_const(Field::from(0x30644e72e131a029b85045b68181585du128));
    let modulus_lo_m1 = b.field_const(Field::from(0x2833e84879b9709143e1f593f0000000u128));
    let two_to_8 = b.field_const(Field::from(256u128));
    let two_to_64 = b.field_const(two_pow(64));
    let two_to_128 = b.field_const(two_pow(128));
    let zero = b.field_const(Field::ZERO);

    let pure_value = b.value_of(value);
    let bytes_arr = b.to_radix(pure_value, Radix::Bytes, Endianness::Big, 32);

    let mut bytes = Vec::with_capacity(32);
    let mut limbs = [zero; 4];
    let mut full_sum = zero;
    for i in 0..31 {
        let idx = b.u_const(32, i as u128);
        let byte = b.array_get(bytes_arr, idx);
        let byte_field = b.cast_to_field(byte);
        let byte_wit = b.write_witness(byte_field);
        b.lookup_rngchk_8(byte_wit, flag);
        bytes.push(byte_wit);

        let limb_idx = i / 8;
        let shifted_limb = b.mul(limbs[limb_idx], two_to_8);
        limbs[limb_idx] = b.add(shifted_limb, byte_wit);

        let shifted_full = b.mul(full_sum, two_to_8);
        full_sum = b.add(shifted_full, byte_wit);
    }

    let full_sum_shifted = b.mul(full_sum, two_to_8);
    let lsb = b.sub(value, full_sum_shifted);
    b.lookup_rngchk_8(lsb, flag);
    bytes.push(lsb);

    let shifted_limb = b.mul(limbs[3], two_to_8);
    limbs[3] = b.add(shifted_limb, lsb);

    let hi_upper = b.mul(limbs[0], two_to_64);
    let hi = b.add(hi_upper, limbs[1]);
    let lo_upper = b.mul(limbs[2], two_to_64);
    let lo = b.add(lo_upper, limbs[3]);

    let limb2_pure = b.value_of(limbs[2]);
    let limb3_pure = b.value_of(limbs[3]);
    let limb2_u64 = b.cast_to(Type::u(64), limb2_pure);
    let limb3_u64 = b.cast_to(Type::u(64), limb3_pure);
    let mod_limb2 = b.u_const(64, 0x2833e84879b97091u64 as u128);
    let mod_limb3 = b.u_const(64, 0x43e1f593f0000000u64 as u128);
    let hi_lt = b.lt(mod_limb2, limb2_u64);
    let hi_eq = b.eq(mod_limb2, limb2_u64);
    let lo_lt = b.lt(mod_limb3, limb3_u64);
    let hi_eq_f = b.cast_to_field(hi_eq);
    let lo_lt_f = b.cast_to_field(lo_lt);
    let hi_eq_and_lo_lt = b.mul(hi_eq_f, lo_lt_f);
    let hi_lt_f = b.cast_to_field(hi_lt);
    let borrow_hint = b.add(hi_lt_f, hi_eq_and_lo_lt);
    let borrow_wit = b.write_witness(borrow_hint);
    b.constrain(borrow_wit, borrow_wit, borrow_wit);

    let borrow_shift = b.mul(borrow_wit, two_to_128);
    let tmp1 = b.sub(modulus_lo_m1, lo);
    let result_lo = b.add(tmp1, borrow_shift);

    let tmp3 = b.sub(modulus_hi, hi);
    let result_hi = b.sub(tmp3, borrow_wit);
    b.rangecheck(result_hi, 128);
    b.rangecheck(result_lo, 128);

    bytes
}

fn lower_field_bit_range_from_bytes(
    b: &mut HLBlockEmitter<'_>,
    bytes: &[ValueId],
    offset: usize,
    width: usize,
    flag: ValueId,
) -> ValueId {
    let low_end = lower_field_low_bits_from_bytes(b, bytes, offset + width, flag);
    let low_start = lower_field_low_bits_from_bytes(b, bytes, offset, flag);
    let selected_shifted = b.sub(low_end, low_start);
    let divisor = b.field_const(two_pow(offset));
    b.div(selected_shifted, divisor)
}

fn lower_field_low_bits_from_bytes(
    b: &mut HLBlockEmitter<'_>,
    bytes: &[ValueId],
    bits: usize,
    flag: ValueId,
) -> ValueId {
    assert!(bits <= 254, "field BitRange exceeds canonical field width");
    if bits == 0 {
        return b.field_const(Field::ZERO);
    }

    let two_to_8 = b.field_const(Field::from(256u128));
    let full_bytes = bits / 8;
    let partial_bits = bits % 8;
    let start = 32 - full_bytes - usize::from(partial_bits > 0);
    let mut value = b.field_const(Field::ZERO);
    for (i, byte) in bytes.iter().enumerate().skip(start) {
        let elem = if i == start && partial_bits > 0 {
            split_partial_field_byte(b, *byte, partial_bits, flag)
        } else {
            *byte
        };
        let shifted = b.mul(value, two_to_8);
        value = b.add(shifted, elem);
    }
    value
}

fn split_partial_field_byte(
    b: &mut HLBlockEmitter<'_>,
    byte_wit: ValueId,
    lo_size: usize,
    flag: ValueId,
) -> ValueId {
    assert!(
        (1..8).contains(&lo_size),
        "partial byte split must be non-empty"
    );
    let hi_size = 8 - lo_size;
    let two_to_lo = b.field_const(Field::from(1u128 << lo_size));

    let byte_pure = b.value_of(byte_wit);
    let byte_u8 = b.cast_to(Type::u(8), byte_pure);
    let divisor = b.u_const(8, 1u128 << lo_size);
    let hi_hint_u8 = b.div(byte_u8, divisor);
    let hi_hint = b.cast_to_field(hi_hint_u8);
    let hi_wit = b.write_witness(hi_hint);

    let hi_bound = b.field_const(Field::from((1u128 << hi_size) - 1));
    let hi_gap = b.sub(hi_bound, hi_wit);
    b.lookup_rngchk_8(hi_gap, flag);

    let hi_shifted = b.mul(hi_wit, two_to_lo);
    let lo = b.sub(byte_wit, hi_shifted);

    let lo_bound = b.field_const(Field::from((1u128 << lo_size) - 1));
    let lo_gap = b.sub(lo_bound, lo);
    b.lookup_rngchk_8(lo_gap, flag);

    lo
}

fn lower_pure_bit_range_value(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    value_type: &Type,
    offset: usize,
    width: usize,
) -> ValueId {
    match value_type.strip_witness().expr {
        TypeExpr::U(bits) | TypeExpr::I(bits) => {
            assert!(
                bits <= MAX_SUPPORTED_UNSIGNED_BITS,
                "pure integer BitRange lowering only supports up to {MAX_SUPPORTED_UNSIGNED_BITS}-bit integers"
            );
            let unsigned = b.cast_to(Type::u(bits), value);
            let mask = b.u_const(bits, bit_mask(bits, offset, width));
            let masked = b.fresh_value();
            b.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::And,
                result: masked,
                lhs: unsigned,
                rhs: mask,
            });
            let divisor = b.u_const(bits, 1u128 << offset);
            b.div(masked, divisor)
        }
        TypeExpr::Field => lower_pure_field_bit_range_value(b, value, offset, width),
        other => panic!("BitRange expects a scalar source, got {:?}", other),
    }
}

fn lower_pure_field_bit_range_value(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    offset: usize,
    width: usize,
) -> ValueId {
    let low_end = lower_pure_field_low_bits(b, value, offset + width);
    let low_start = lower_pure_field_low_bits(b, value, offset);
    let selected_shifted = b.sub(low_end, low_start);
    let divisor = b.field_const(two_pow(offset));
    b.div(selected_shifted, divisor)
}

fn lower_pure_field_low_bits(b: &mut HLBlockEmitter<'_>, value: ValueId, bits: usize) -> ValueId {
    assert!(bits <= 254, "field BitRange exceeds canonical field width");
    if bits == 0 {
        return b.field_const(Field::ZERO);
    }

    let bytes_arr = b.to_radix(value, Radix::Bytes, Endianness::Big, 32);
    let two_to_8 = b.field_const(Field::from(256u128));
    let full_bytes = bits / 8;
    let partial_bits = bits % 8;
    let start = 32 - full_bytes - usize::from(partial_bits > 0);
    let mut result = b.field_const(Field::ZERO);
    for i in start..32 {
        let idx = b.u_const(32, i as u128);
        let byte = b.array_get(bytes_arr, idx);
        let byte = if i == start && partial_bits > 0 {
            lower_pure_byte_low_bits(b, byte, partial_bits)
        } else {
            byte
        };
        let byte_field = b.cast_to_field(byte);
        let shifted = b.mul(result, two_to_8);
        result = b.add(shifted, byte_field);
    }
    result
}

fn lower_pure_byte_low_bits(b: &mut HLBlockEmitter<'_>, byte: ValueId, bits: usize) -> ValueId {
    assert!(
        (1..8).contains(&bits),
        "partial byte width must be non-empty"
    );
    let divisor = b.u_const(8, 1u128 << bits);
    let high = b.div(byte, divisor);
    let high_shifted = b.mul(high, divisor);
    b.sub(byte, high_shifted)
}

fn two_pow(exponent: usize) -> Field {
    Field::from(2).pow([exponent as u64])
}

fn bit_mask(bits: usize, offset: usize, width: usize) -> u128 {
    assert!(
        bits <= MAX_SUPPORTED_UNSIGNED_BITS,
        "u{MAX_SUPPORTED_UNSIGNED_BITS} mask cannot represent u{bits}"
    );
    assert!(width > 0, "BitRange width must be at least 1");
    assert!(offset + width <= bits, "BitRange exceeds source width");
    if width == MAX_SUPPORTED_UNSIGNED_BITS {
        u128::MAX
    } else {
        ((1u128 << width) - 1) << offset
    }
}

fn cast_target_for_scalar_type(ty: &Type) -> Type {
    match ty.strip_witness().expr {
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => ty.clone(),
        other => panic!("BitRange result must be scalar, got {:?}", other),
    }
}
