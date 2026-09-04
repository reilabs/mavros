//! Lowers canonical `BitRange` operations after the witness integer/bitwise passes have emitted
//! all bit selections.

use mavros_int_semantics::{IntBits, int_bits::FIELD_LIMB_BITS};

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, Endianness, OpCode, Radix, Type, TypeExpr,
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
        let source_bits = value_type.get_bit_size(b.field());
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
        let source_bits = value_type.get_bit_size(b.field());
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

        let mut reconstructed = low.unwrap_or_else(|| b.field_const(b.field().zero()));
        let result_shift = b.field_const(b.field().two_pow(offset));
        let result_shifted = b.umul(result_field, result_shift);
        reconstructed = b.uadd(reconstructed, result_shifted);
        if let Some(high) = high {
            let high_shift = b.field_const(b.field().two_pow(offset + width));
            let high_shifted = b.umul(high, high_shift);
            reconstructed = b.uadd(reconstructed, high_shifted);
        }

        let value_field = b.cast_to_field(value);
        let diff = b.usub(value_field, reconstructed);
        let zero = b.field_const(b.field().zero());
        let flag = b.field_const(b.field().one());
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
        let flag = b.field_const(b.field().one());
        let bytes = decompose_canonical_field_bytes(b, value, flag);
        let selected = lower_field_bit_range_from_bytes(b, &bytes, offset, width, flag);

        b.emit(OpCode::Cast {
            result,
            value: selected,
            target: CastTarget::Field,
        });
    }
}

// FIELD-ASSUMPTION: L3-felt-limbs
// The modulus value is read from the configured field (see below); the residual assumption is
// structural in that the field fits in 4 limbs / 32 bytes for the canonical byte-decomposition.
fn decompose_canonical_field_bytes(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    flag: ValueId,
) -> Vec<ValueId> {
    // The field modulus, split into its high and low 128-bit halves, drives the canonical
    // byte-decomposition below. Read from the configured field (not hardcoded), so no concrete
    // prime is named here; `modulus_lo_m1` is the low half minus one.
    let modulus_limbs = b.field().modulus_limbs();
    let modulus_hi_u128 =
        (u128::from(modulus_limbs[3]) << FIELD_LIMB_BITS) | u128::from(modulus_limbs[2]);
    let modulus_lo_u128 =
        (u128::from(modulus_limbs[1]) << FIELD_LIMB_BITS) | u128::from(modulus_limbs[0]);

    // `p` is odd, so subtracting one never borrows out of the lowest limb.
    let modulus_lo_m1_u128 = modulus_lo_u128 - 1;
    let modulus_hi = b.field_const(b.field().constant(modulus_hi_u128));
    let modulus_lo_m1 = b.field_const(b.field().constant(modulus_lo_m1_u128));
    let two_to_8 = b.field_const(b.field().constant(256u128));
    // One field limb's place value, and one `u128` half's. The `128`s here and below are the
    // width of the halves the modulus is read in above, not a field limb.
    let limb_place_value = b.field_const(b.field().two_pow(FIELD_LIMB_BITS));
    let two_to_128 = b.field_const(b.field().two_pow(128));
    let zero = b.field_const(b.field().zero());

    let pure_value = b.value_of(value);
    let bytes_arr = b.to_radix(pure_value, Radix::Bytes, Endianness::Big, 32);

    let mut bytes = Vec::with_capacity(32);
    let mut limbs = [zero; 4];
    let mut full_sum = zero;
    for i in 0..31 {
        let idx = b.int_const(IntBits::from_u128(32, i as u128));
        let byte = b.array_get(bytes_arr, idx);
        let byte_field = b.cast_to_field(byte);
        let byte_wit = b.write_witness(byte_field);
        b.lookup_rngchk_8(byte_wit, flag);
        bytes.push(byte_wit);

        let limb_idx = i / 8;
        let shifted_limb = b.umul(limbs[limb_idx], two_to_8);
        limbs[limb_idx] = b.uadd(shifted_limb, byte_wit);

        let shifted_full = b.umul(full_sum, two_to_8);
        full_sum = b.uadd(shifted_full, byte_wit);
    }

    let full_sum_shifted = b.umul(full_sum, two_to_8);
    let lsb = b.usub(value, full_sum_shifted);
    b.lookup_rngchk_8(lsb, flag);
    bytes.push(lsb);

    let shifted_limb = b.umul(limbs[3], two_to_8);
    limbs[3] = b.uadd(shifted_limb, lsb);

    let hi_upper = b.umul(limbs[0], limb_place_value);
    let hi = b.uadd(hi_upper, limbs[1]);
    let lo_upper = b.umul(limbs[2], limb_place_value);
    let lo = b.uadd(lo_upper, limbs[3]);

    let limb2_pure = b.value_of(limbs[2]);
    let limb3_pure = b.value_of(limbs[3]);
    let limb2_u64 = b.cast_to(CastTarget::Int(FIELD_LIMB_BITS), limb2_pure);
    let limb3_u64 = b.cast_to(CastTarget::Int(FIELD_LIMB_BITS), limb3_pure);

    // The two 64-bit limbs of the low half of `p - 1`, in the same big-endian limb order the byte
    // decomposition above produces; derived from the configured field rather than written out.
    let mod_limb2 = b.int_const(IntBits::from_u128(
        FIELD_LIMB_BITS,
        u128::from((modulus_lo_m1_u128 >> FIELD_LIMB_BITS) as u64),
    ));
    let mod_limb3 = b.int_const(IntBits::from_u128(
        FIELD_LIMB_BITS,
        u128::from(modulus_lo_m1_u128 as u64),
    ));
    let hi_lt = b.ult(mod_limb2, limb2_u64);
    let hi_eq = b.eq(mod_limb2, limb2_u64);
    let lo_lt = b.ult(mod_limb3, limb3_u64);
    let hi_eq_f = b.cast_to_field(hi_eq);
    let lo_lt_f = b.cast_to_field(lo_lt);
    let hi_eq_and_lo_lt = b.umul(hi_eq_f, lo_lt_f);
    let hi_lt_f = b.cast_to_field(hi_lt);
    let borrow_hint = b.uadd(hi_lt_f, hi_eq_and_lo_lt);
    let borrow_wit = b.write_witness(borrow_hint);
    b.constrain(borrow_wit, borrow_wit, borrow_wit);

    let borrow_shift = b.umul(borrow_wit, two_to_128);
    let tmp1 = b.usub(modulus_lo_m1, lo);
    let result_lo = b.uadd(tmp1, borrow_shift);

    let tmp3 = b.usub(modulus_hi, hi);
    let result_hi = b.usub(tmp3, borrow_wit);
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
    let selected_shifted = b.usub(low_end, low_start);
    let divisor = b.field_const(b.field().two_pow(offset));
    b.udiv(selected_shifted, divisor)
}

fn lower_field_low_bits_from_bytes(
    b: &mut HLBlockEmitter<'_>,
    bytes: &[ValueId],
    bits: usize,
    flag: ValueId,
) -> ValueId {
    assert!(
        bits <= b.field().field_bit_size() as usize,
        "field BitRange exceeds canonical field width"
    );
    if bits == 0 {
        return b.field_const(b.field().zero());
    }

    let two_to_8 = b.field_const(b.field().constant(256u128));
    let full_bytes = bits / 8;
    let partial_bits = bits % 8;
    let start = 32 - full_bytes - usize::from(partial_bits > 0);
    let mut value = b.field_const(b.field().zero());
    for (i, byte) in bytes.iter().enumerate().skip(start) {
        let elem = if i == start && partial_bits > 0 {
            split_partial_field_byte(b, *byte, partial_bits, flag)
        } else {
            *byte
        };
        let shifted = b.umul(value, two_to_8);
        value = b.uadd(shifted, elem);
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
    let two_to_lo = b.field_const(b.field().constant(1u128 << lo_size));

    let byte_pure = b.value_of(byte_wit);
    let byte_u8 = b.cast_to(CastTarget::Int(8), byte_pure);
    let divisor = b.int_const(IntBits::from_u128(8, 1u128 << lo_size));
    let hi_hint_u8 = b.udiv(byte_u8, divisor);
    let hi_hint = b.cast_to_field(hi_hint_u8);
    let hi_wit = b.write_witness(hi_hint);

    let hi_bound = b.field_const(b.field().constant((1u128 << hi_size) - 1));
    let hi_gap = b.usub(hi_bound, hi_wit);
    b.lookup_rngchk_8(hi_gap, flag);

    let hi_shifted = b.umul(hi_wit, two_to_lo);
    let lo = b.usub(byte_wit, hi_shifted);

    let lo_bound = b.field_const(b.field().constant((1u128 << lo_size) - 1));
    let lo_gap = b.usub(lo_bound, lo);
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
        TypeExpr::Int(bits) => {
            let unsigned = b.cast_to(CastTarget::Int(bits), value);
            let mask = b.int_const(window_mask(bits, offset, width));
            let masked = b.fresh_value();
            b.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::And,
                result: masked,
                lhs: unsigned,
                rhs: mask,
            });
            let divisor = b.two_pow_const(bits, offset);
            b.udiv(masked, divisor)
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
    let selected_shifted = b.usub(low_end, low_start);
    let divisor = b.field_const(b.field().two_pow(offset));
    b.udiv(selected_shifted, divisor)
}

fn lower_pure_field_low_bits(b: &mut HLBlockEmitter<'_>, value: ValueId, bits: usize) -> ValueId {
    assert!(
        bits <= b.field().field_bit_size() as usize,
        "field BitRange exceeds canonical field width"
    );
    if bits == 0 {
        return b.field_const(b.field().zero());
    }

    let bytes_arr = b.to_radix(value, Radix::Bytes, Endianness::Big, 32);
    let two_to_8 = b.field_const(b.field().constant(256u128));
    let full_bytes = bits / 8;
    let partial_bits = bits % 8;
    let start = 32 - full_bytes - usize::from(partial_bits > 0);
    let mut result = b.field_const(b.field().zero());
    for i in start..32 {
        let idx = b.int_const(IntBits::from_u128(32, i as u128));
        let byte = b.array_get(bytes_arr, idx);
        let byte = if i == start && partial_bits > 0 {
            lower_pure_byte_low_bits(b, byte, partial_bits)
        } else {
            byte
        };
        let byte_field = b.cast_to_field(byte);
        let shifted = b.umul(result, two_to_8);
        result = b.uadd(shifted, byte_field);
    }
    result
}

fn lower_pure_byte_low_bits(b: &mut HLBlockEmitter<'_>, byte: ValueId, bits: usize) -> ValueId {
    assert!(
        (1..8).contains(&bits),
        "partial byte width must be non-empty"
    );
    let divisor = b.int_const(IntBits::from_u128(8, 1u128 << bits));
    let high = b.udiv(byte, divisor);
    let high_shifted = b.umul(high, divisor);
    b.usub(byte, high_shifted)
}

/// The `width` bits starting at `offset` of a `bits`-wide value, as a mask of that width.
///
/// A **pattern** rather than a host word, which is what makes the window width-generic: the mask
/// of a 200-bit window is not a number any host type has, and building it through one is what used
/// to stop this lowering at [`narrow_int_bits`]. The only bounds left are the window's own — a
/// window has at least one bit and does not reach past its source.
///
/// Named for the window rather than for the mask so as not to collide with [`util::bit_mask`],
/// which answers the different question "the low `bits` of a host word" and is what the guard
/// lowerings reach for.
///
/// [`util::bit_mask`]: crate::compiler::util::bit_mask
fn window_mask(bits: usize, offset: usize, width: usize) -> IntBits {
    assert!(width > 0, "BitRange width must be at least 1");
    assert!(offset + width <= bits, "BitRange exceeds source width");
    IntBits::all_ones(width).cast(bits).shifted_left(offset)
}

fn cast_target_for_scalar_type(ty: &Type) -> CastTarget {
    match ty.strip_witness().expr {
        TypeExpr::Field => CastTarget::Field,
        // A `CastTarget` is a raw-bits conversion, so there is one target per width and no sign to
        // choose: `TypeExpr::Int(n)` says only "an n-bit integer", and `CastTarget::Int(n)` says
        // only "reinterpret at n bits". Sign extension is the separate `SExt` opcode.
        TypeExpr::Int(bits) => CastTarget::Int(bits),
        other => panic!("BitRange result must be scalar, got {:?}", other),
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask is a pattern, so a window past every host type is an ordinary one.
    ///
    /// This is the whole of what makes the window width-generic: `((1u128 << width) - 1) << offset`
    /// has no answer at 200 bits, and the bound that used to be written against it is what stopped
    /// a witnessed narrowing at `narrow_int_bits`.
    #[test]
    fn a_window_past_every_host_type_is_an_ordinary_mask() {
        let mask = window_mask(200, 64, 100);

        assert_eq!(mask.bits(), 200);
        for bit in [63usize, 164, 199] {
            assert_eq!(
                mask.bit(bit),
                Some(false),
                "bit {bit} is outside the window"
            );
        }
        for bit in [64usize, 100, 163] {
            assert_eq!(mask.bit(bit), Some(true), "bit {bit} is inside the window");
        }
    }

    /// The window's own bounds, which are the only ones left.
    #[test]
    #[should_panic(expected = "BitRange exceeds source width")]
    fn a_window_reaching_past_its_source_is_refused_before_it_can_truncate() {
        let _ = window_mask(128, 100, 100);
    }

    #[test]
    #[should_panic(expected = "BitRange width must be at least 1")]
    fn an_empty_window_is_refused() {
        let _ = window_mask(128, 0, 0);
    }

    /// A whole-width window is every bit, at a width a host word still reaches and at one it does
    /// not.
    #[test]
    fn a_whole_width_window_is_all_ones() {
        assert!(window_mask(128, 0, 128).is_all_ones());
        assert!(window_mask(253, 0, 253).is_all_ones());
    }
}
