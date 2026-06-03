//! Lowers canonical `BitRange` operations after the witness integer/bitwise passes have emitted
//! all bit selections.

use ark_ff::{AdditiveGroup as _, Field as _};

use crate::compiler::{
    Field,
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, Endianness, MAX_SUPPORTED_UNSIGNED_BITS, OpCode, Radix,
            Type, TypeExpr,
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

        if offset == 0 {
            self.lower_witness_low_bit_range(b, context, result, value, width, source_bits);
            return;
        }

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

        let result_field = b.cast_to_field(result);
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

        let value_field = b.cast_to_field(value);
        let diff = b.sub(value_field, reconstructed);
        let zero = b.field_const(Field::ZERO);
        let flag = b.field_const(Field::ONE);
        b.constrain(flag, diff, zero);
    }

    fn lower_witness_low_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        width: usize,
        source_bits: usize,
    ) {
        let value_type = context.types().get_value_type(value);
        let value_field = b.cast_to_field(value);
        let high_bits = source_bits - width;
        let low = if high_bits == 0 {
            value_field
        } else {
            let pure_value = b.value_of(value);
            let high_hint = lower_pure_bit_range_value(
                b,
                pure_value,
                &value_type.strip_witness(),
                width,
                high_bits,
            );
            let high_hint = b.cast_to_field(high_hint);
            let high = b.write_witness(high_hint);
            b.rangecheck(high, high_bits);

            let shift = b.field_const(two_pow(width));
            let high_shifted = b.mul(high, shift);
            b.sub(value_field, high_shifted)
        };

        b.rangecheck(low, width);
        b.emit(OpCode::Cast {
            result,
            value: low,
            target: cast_target_for_scalar_type(context.types().get_value_type(result)),
        });
    }

    fn lower_witness_field_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    ) {
        let limbs = decompose_field_limbs(b, value);
        let selection =
            select_field_bits_from_limbs(b, limbs.lo_u128, limbs.hi_u128, offset, width);

        if !selection.lo_rangechecked {
            b.rangecheck(limbs.lo, 128);
        }
        if !selection.hi_rangechecked {
            b.rangecheck(limbs.hi, 128);
        }

        assert_field_canonical(b, limbs.lo, limbs.hi, limbs.two_128);

        b.emit(OpCode::Cast {
            result,
            value: selection.value,
            target: CastTarget::Field,
        });
    }
}

/// High 126 bits of the BN254 scalar modulus `p` (equivalently, of `p - 1`).
const MODULUS_HI: u128 = 0x30644e72e131a029b85045b68181585d;
/// Low 128 bits of `p - 1`, so that `p - 1 = MODULUS_HI * 2^128 + MODULUS_M1_LO`.
const MODULUS_M1_LO: u128 = 0x2833e84879b9709143e1f593f0000000;

struct FieldLimbs {
    lo: ValueId,
    hi: ValueId,
    lo_u128: ValueId,
    hi_u128: ValueId,
    two_128: ValueId,
}

/// Split a `WitnessOf<Field>` into `lo` (low 128 bits) and `hi` limbs. The caller must prove both
/// limbs are range-checked, either through bit extraction or through standalone rangechecks, then
/// assert canonicality with `assert_field_canonical`.
fn decompose_field_limbs(b: &mut HLBlockEmitter<'_>, value: ValueId) -> FieldLimbs {
    let two_128 = b.field_const(two_pow(128));

    let hi = witness_high_limb(b, value, two_128);
    let lo = recover_low_limb(b, value, hi, two_128);

    let lo_u128 = b.cast_to(CastTarget::U(128), lo);
    let hi_u128 = b.cast_to(CastTarget::U(128), hi);
    FieldLimbs {
        lo,
        hi,
        lo_u128,
        hi_u128,
        two_128,
    }
}

fn witness_high_limb(b: &mut HLBlockEmitter<'_>, value: ValueId, two_128: ValueId) -> ValueId {
    let value = b.value_of(value);
    let low = b.cast_to(CastTarget::U(128), value);
    let low = b.cast_to_field(low);
    let high = b.sub(value, low);
    let high = b.div(high, two_128);
    b.write_witness(high)
}

fn recover_low_limb(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    hi: ValueId,
    two_128: ValueId,
) -> ValueId {
    let value = b.cast_to_field(value);
    let hi_shifted = b.mul(hi, two_128);
    b.sub(value, hi_shifted)
}

/// Assert `lo + hi * 2^128 <= p - 1` (i.e. the limbs are the canonical representation), by
/// performing the 254-bit subtraction `(p - 1) - value` with a hinted borrow and range-checking
/// that both resulting limbs are non-negative.
fn assert_field_canonical(b: &mut HLBlockEmitter<'_>, lo: ValueId, hi: ValueId, two_128: ValueId) {
    let lo_pure = b.value_of(lo);
    let lo_pure = b.cast_to(CastTarget::U(128), lo_pure);
    let modulus_m1_lo_u = b.u_const(128, MODULUS_M1_LO);
    let borrow_bool = b.lt(modulus_m1_lo_u, lo_pure);
    let borrow_hint = b.cast_to_field(borrow_bool);
    let borrow = b.write_witness(borrow_hint);
    b.constrain(borrow, borrow, borrow);

    let modulus_m1_lo = b.field_const(Field::from(MODULUS_M1_LO));
    let modulus_hi = b.field_const(Field::from(MODULUS_HI));
    let borrow_shifted = b.mul(borrow, two_128);

    let lo_gap = b.sub(modulus_m1_lo, lo);
    let lo_gap = b.add(lo_gap, borrow_shifted);
    let hi_gap = b.sub(modulus_hi, hi);
    let hi_gap = b.sub(hi_gap, borrow);
    b.rangecheck(lo_gap, 128);
    b.rangecheck(hi_gap, 128);
}

struct FieldBitSelection {
    value: ValueId,
    lo_rangechecked: bool,
    hi_rangechecked: bool,
}

/// Select bits `[offset, offset + width)` from the 128-bit `lo` limb and high `hi` limb. A selected
/// limb is range-checked by the integer `BitRange` proof, so the caller can skip its standalone
/// limb rangecheck.
fn select_field_bits_from_limbs(
    b: &mut HLBlockEmitter<'_>,
    lo: ValueId,
    hi: ValueId,
    offset: usize,
    width: usize,
) -> FieldBitSelection {
    const SPLIT: usize = 128;
    if offset + width <= SPLIT {
        let bits = b.bit_range(lo, offset, width);
        FieldBitSelection {
            value: b.cast_to_field(bits),
            lo_rangechecked: true,
            hi_rangechecked: false,
        }
    } else if offset >= SPLIT {
        let bits = b.bit_range(hi, offset - SPLIT, width);
        FieldBitSelection {
            value: b.cast_to_field(bits),
            lo_rangechecked: false,
            hi_rangechecked: true,
        }
    } else {
        let lo_width = SPLIT - offset;
        let hi_width = offset + width - SPLIT;
        let lo_bits = b.bit_range(lo, offset, lo_width);
        let hi_bits = b.bit_range(hi, 0, hi_width);
        let lo_field = b.cast_to_field(lo_bits);
        let hi_field = b.cast_to_field(hi_bits);
        let shift = b.field_const(two_pow(lo_width));
        let hi_shifted = b.mul(hi_field, shift);
        FieldBitSelection {
            value: b.add(lo_field, hi_shifted),
            lo_rangechecked: true,
            hi_rangechecked: true,
        }
    }
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
            let unsigned = b.cast_to(CastTarget::U(bits), value);
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

fn cast_target_for_scalar_type(ty: &Type) -> CastTarget {
    match ty.strip_witness().expr {
        TypeExpr::Field => CastTarget::Field,
        TypeExpr::U(bits) => CastTarget::U(bits),
        TypeExpr::I(bits) => CastTarget::I(bits),
        other => panic!("BitRange result must be scalar, got {:?}", other),
    }
}
