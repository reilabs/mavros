//! Lowers canonical `BitRange` operations after the witness integer/bitwise passes have emitted
//! all bit selections.

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{IntBits, int_bits::FIELD_LIMB_BITS};
use num_bigint::BigUint;

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

/// The shape of an element's canonical byte decomposition: [`Self::bytes`] big-endian bytes, whose
/// integer splits at bit [`Self::split`] into the two halves compared against `p - 1`'s.
///
/// Each half's distance below `p - 1`'s, with the borrow between them, is range-checked to `split`
/// bits. The integer distances lie in `[-2^split, 2^(split + 1))`, as the high half is no wider
/// than the low, so a range check reads them as integers only for `p > 2^(split + 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalBytes {
    /// One byte for every eight bits of the modulus, rounded up.
    bytes: usize,

    /// The width of the low half, in bits: the larger half of the bytes.
    split: usize,
}

impl CanonicalBytes {
    /// The decomposition of an element of `field`.
    ///
    /// # Panics
    ///
    /// If the modulus is too small for a range check to read the distances as integers.
    fn of(field: FieldConfig) -> Self {
        let bytes = (field.field_bit_size() as usize).div_ceil(8);
        let split = 8 * bytes.div_ceil(2);
        assert!(
            BigUint::from(field.modulus()) > BigUint::from(1u8) << (split + 1),
            "a canonical byte decomposition splits at bit {split}, which needs a modulus above \
             2^{}",
            split + 1
        );
        Self { bytes, split }
    }
}

// FIELD-ASSUMPTION: L3-felt-limbs
// The modulus is read through its four-limb `BigInt`, and each half into a `u128`, so the field
// fits 256 bits.
/// Decomposes the witnessed field element `value` into its canonical big-endian bytes, each
/// range-checked, and returns their witnesses.
///
/// The bytes are the prover's hint, and the constraints hold only for those of the element's
/// canonical integer: the bytes of `value + p`, which sum to the same element, are refused.
fn decompose_canonical_field_bytes(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    flag: ValueId,
) -> Vec<ValueId> {
    let CanonicalBytes {
        bytes: count,
        split,
    } = CanonicalBytes::of(b.field());
    let modulus = BigUint::from(b.field().modulus());
    let half = |half: BigUint| u128::try_from(half).expect("a half of the modulus fits a u128");
    let modulus_hi_u128 = half(&modulus >> split);
    // `p` is odd, so subtracting one never borrows out of the low half.
    let modulus_lo_m1_u128 = half(modulus & ((BigUint::from(1u8) << split) - 1u8)) - 1;
    let modulus_hi = b.field_const(b.field().constant(modulus_hi_u128));
    let modulus_lo_m1 = b.field_const(b.field().constant(modulus_lo_m1_u128));
    let two_to_8 = b.field_const(b.field().constant(256u128));
    let limb_place_value = b.field_const(b.field().two_pow(FIELD_LIMB_BITS));
    let two_to_split = b.field_const(b.field().two_pow(split));
    let zero = b.field_const(b.field().zero());

    let pure_value = b.value_of(value);
    let bytes_arr = b.to_radix(pure_value, Radix::Bytes, Endianness::Big, count);

    // Each half is summed in limbs of up to eight bytes, counted from its least significant byte,
    // which the borrow's hint compares as host words. The limbs are in big-endian order.
    let limb_bytes = FIELD_LIMB_BITS / 8;
    let lo_bytes = split / 8;
    let hi_limbs = (count - lo_bytes).div_ceil(limb_bytes);
    let lo_limbs = lo_bytes.div_ceil(limb_bytes);
    let limb_of = |byte: usize| {
        let place = count - 1 - byte;
        if place < lo_bytes {
            hi_limbs + lo_limbs - 1 - place / limb_bytes
        } else {
            hi_limbs - 1 - (place - lo_bytes) / limb_bytes
        }
    };

    let mut bytes = Vec::with_capacity(count);
    let mut limbs = vec![zero; hi_limbs + lo_limbs];
    let mut full_sum = zero;
    for i in 0..count - 1 {
        let idx = b.int_const(IntBits::from_u128(32, i as u128));
        let byte = b.array_get(bytes_arr, idx);
        let byte_field = b.cast_to_field(byte);
        let byte_wit = b.write_witness(byte_field);
        b.lookup_rngchk_8(byte_wit, flag);
        bytes.push(byte_wit);

        let limb_idx = limb_of(i);
        let shifted_limb = b.umul(limbs[limb_idx], two_to_8);
        limbs[limb_idx] = b.uadd(shifted_limb, byte_wit);

        let shifted_full = b.umul(full_sum, two_to_8);
        full_sum = b.uadd(shifted_full, byte_wit);
    }

    let full_sum_shifted = b.umul(full_sum, two_to_8);
    let lsb = b.usub(value, full_sum_shifted);
    b.lookup_rngchk_8(lsb, flag);
    bytes.push(lsb);

    let last = limbs.len() - 1;
    let shifted_limb = b.umul(limbs[last], two_to_8);
    limbs[last] = b.uadd(shifted_limb, lsb);

    let (hi_limbs, lo_limbs) = limbs.split_at(hi_limbs);
    let hi = sum_limbs(b, hi_limbs, limb_place_value);
    let lo = sum_limbs(b, lo_limbs, limb_place_value);

    // The borrow's hint: whether the low half exceeds `p - 1`'s, compared limb by limb from the top.
    let lo_pure: Vec<_> = lo_limbs.iter().map(|&limb| b.value_of(limb)).collect();
    let lo_u64: Vec<_> = lo_pure
        .iter()
        .map(|&limb| b.cast_to(CastTarget::Int(FIELD_LIMB_BITS), limb))
        .collect();
    let mod_limbs: Vec<_> = (0..lo_limbs.len())
        .rev()
        .map(|place| {
            let limb = (modulus_lo_m1_u128 >> (FIELD_LIMB_BITS * place)) as u64;
            b.int_const(IntBits::from_u128(FIELD_LIMB_BITS, u128::from(limb)))
        })
        .collect();
    let mut exceeds = Vec::with_capacity(lo_limbs.len());
    let mut equals = Vec::with_capacity(lo_limbs.len() - 1);
    for (i, (&modulus_limb, &limb)) in mod_limbs.iter().zip(&lo_u64).enumerate() {
        exceeds.push(b.ult(modulus_limb, limb));
        if i + 1 < lo_limbs.len() {
            equals.push(b.eq(modulus_limb, limb));
        }
    }
    // `exceeds_i + equals_i * (the borrow of the limbs below)`, folded up from the bottom limb.
    let last_exceeds = exceeds[exceeds.len() - 1];
    let mut borrow = None;
    for (&exceeds_here, &equals_here) in exceeds.iter().zip(&equals).rev() {
        let equals_f = b.cast_to_field(equals_here);
        let below = borrow.unwrap_or_else(|| b.cast_to_field(last_exceeds));
        let carried = b.umul(equals_f, below);
        let exceeds_f = b.cast_to_field(exceeds_here);
        borrow = Some(b.uadd(exceeds_f, carried));
    }
    let borrow_hint = borrow.unwrap_or_else(|| b.cast_to_field(last_exceeds));
    let borrow_wit = b.write_witness(borrow_hint);
    b.constrain(borrow_wit, borrow_wit, borrow_wit);

    let borrow_shift = b.umul(borrow_wit, two_to_split);
    let tmp1 = b.usub(modulus_lo_m1, lo);
    let result_lo = b.uadd(tmp1, borrow_shift);

    let tmp3 = b.usub(modulus_hi, hi);
    let result_hi = b.usub(tmp3, borrow_wit);
    b.rangecheck(result_hi, split);
    b.rangecheck(result_lo, split);

    bytes
}

/// The integer of big-endian `limbs`, each worth `place` times the next.
fn sum_limbs(b: &mut HLBlockEmitter<'_>, limbs: &[ValueId], place: ValueId) -> ValueId {
    let (&top, rest) = limbs.split_first().expect("a half has at least one limb");
    rest.iter().fold(top, |sum, &limb| {
        let upper = b.umul(sum, place);
        b.uadd(upper, limb)
    })
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
    let start = bytes.len() - full_bytes - usize::from(partial_bits > 0);
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

    // Each gap is below 256 only, which leaves `lo` as low as `2^lo_size - 256`: the prover could
    // raise `hi` and push `lo` negative. Checking `lo` itself pins it to `[0, 2^lo_size)`, and with
    // it `hi`.
    let lo_bound = b.field_const(b.field().constant((1u128 << lo_size) - 1));
    let lo_gap = b.usub(lo_bound, lo);
    b.lookup_rngchk_8(lo_gap, flag);
    b.lookup_rngchk_8(lo, flag);

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
        other => ice!("BitRange expects a scalar source, got {:?}", other),
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

    let count = CanonicalBytes::of(b.field()).bytes;
    let bytes_arr = b.to_radix(value, Radix::Bytes, Endianness::Big, count);
    let two_to_8 = b.field_const(b.field().constant(256u128));
    let full_bytes = bits / 8;
    let partial_bits = bits % 8;
    let start = count - full_bytes - usize::from(partial_bits > 0);
    let mut result = b.field_const(b.field().zero());
    for i in start..count {
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
/// Built as a **pattern**, not a host word, so it works at any width: the mask of a 200-bit window
/// fits no host type, and building it through one would limit this lowering to
/// [`narrow_int_bits`]. The only bounds are on the window: it has at least one bit and does not
/// reach past its source.
///
/// Named for the window, not the mask, to avoid colliding with [`util::bit_mask`], which gives the
/// low `bits` of a host word and is what the guard lowerings use.
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
        other => ice!("BitRange result must be scalar, got {:?}", other),
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use mavros_artifacts::{Field, Goldilocks, InputValueOrdered, MavrosField};

    use super::*;
    use crate::{
        compiler::ssa::{
            SourceLocation,
            hlssa::{Constant, HLSSA, SequenceTargetType, builder::HLSSABuilder},
        },
        forgery::{self, all_ops, replace_hints},
    };

    /// The mask is a pattern, not a host word, so a window may be wider than every host type. A
    /// `u128` mask, `((1u128 << width) - 1) << offset`, has no answer at 200 bits and would cap a
    /// witnessed narrowing at `narrow_int_bits`.
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

    /// The only bounds are on the window: it has at least one bit and does not reach past its
    /// source.
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

    // THE CANONICAL BYTES
    // --------------------------------------------------------------------------------------

    /// bn254 has 32 bytes split at bit 128. Goldilocks has `2^33 < p`, so its 8 bytes split at bit
    /// 32.
    #[test]
    fn each_field_splits_its_bytes_near_the_middle() {
        assert_eq!(
            CanonicalBytes::of(FieldConfig::bn254()),
            CanonicalBytes {
                bytes: 32,
                split: 128
            }
        );
        assert_eq!(
            CanonicalBytes::of(FieldConfig::goldilocks()),
            CanonicalBytes {
                bytes: 8,
                split: 32
            }
        );
    }

    /// Every representative `x + k·p` that fits the bytes sums to the element `x`, so only the
    /// comparison against `p - 1` can refuse it. The test plays a dishonest prover by replacing the
    /// hints after lowering has emitted them; the canonical bytes, put in the same way, are
    /// accepted.
    ///
    /// With the borrow forged too, every representative passes both distance checks, so only the
    /// borrow's boolean constraint refuses it: `borrow = p_hi - hi - d`, where `d·2^split + t` is
    /// `p - 1 - x`, makes the high distance `d` and the low one `t`.
    #[test]
    fn only_the_canonical_bytes_decompose_a_witnessed_element() {
        check_only_the_canonical_bytes_decompose::<Field>(FieldConfig::bn254());
        check_only_the_canonical_bytes_decompose::<Goldilocks>(FieldConfig::goldilocks());
    }

    fn check_only_the_canonical_bytes_decompose<F: MavrosField>(field: FieldConfig) {
        let CanonicalBytes { bytes, split } = CanonicalBytes::of(field);
        let modulus = BigUint::from(field.modulus());
        let room = BigUint::from(1u8) << (8 * bytes);
        let two_to_split = BigUint::from(1u8) << split;
        let largest_with_a_second = (&room - &modulus - 1u8).min(&modulus - 1u8);
        // `2^split - 1` has a low half above `p - 1`'s, so its honest borrow is 1.
        let values = [
            BigUint::ZERO,
            BigUint::from(1u8),
            &two_to_split - 1u8,
            &modulus - 1u8,
            largest_with_a_second,
        ];
        let accepts = |value: &BigUint, forgery: &Forgery| {
            accepts_the_forgery::<F>(field, value, (0, 8), forgery)
        };
        for value in &values {
            let mut representative = value.clone();
            while representative < room {
                let canonical = representative == *value;
                let distance = (&modulus - 1u8 - value) / &two_to_split;
                let borrow =
                    (&modulus + (&modulus >> split) - (&representative >> split) - distance)
                        % &modulus;
                for forgery in [
                    Forgery {
                        bytes: Some(representative.clone()),
                        ..Forgery::default()
                    },
                    Forgery {
                        bytes: Some(representative.clone()),
                        borrow: Some(borrow),
                        ..Forgery::default()
                    },
                ] {
                    assert_eq!(
                        accepts(value, &forgery),
                        canonical,
                        "{:?}: the bytes of {representative} as {value}, with {forgery:?}",
                        F::ID
                    );
                }
                representative += &modulus;
            }

            // The canonical bytes with the other borrow: one distance leaves its range.
            let honest = value & (&two_to_split - 1u8) > (&modulus - 1u8) & (&two_to_split - 1u8);
            let flipped = Forgery {
                borrow: Some(BigUint::from(u8::from(!honest))),
                ..Forgery::default()
            };
            assert!(
                !accepts(value, &flipped),
                "{:?}: {value} is accepted with the borrow {}",
                F::ID,
                u8::from(!honest)
            );
        }
    }

    /// A window ending inside a byte splits that byte, and the split's high part is a hint. Each
    /// way of forging it is refused by one check: raising it pushes the low part negative,
    /// which only the check on the low part refuses; lowering it lifts the low part past its bits,
    /// which only the low gap refuses; and an element off the integers sets the low part to any
    /// value in range, which only the high gap refuses.
    #[test]
    fn a_split_byte_takes_only_its_honest_high_part() {
        check_a_split_byte_takes_only_its_honest_high_part::<Field>(FieldConfig::bn254());
        check_a_split_byte_takes_only_its_honest_high_part::<Goldilocks>(FieldConfig::goldilocks());
    }

    fn check_a_split_byte_takes_only_its_honest_high_part<F: MavrosField>(field: FieldConfig) {
        let modulus = BigUint::from(field.modulus());
        let half: BigUint = (&modulus + 1u8) >> 1u32;
        // Each window has one end inside a byte: the low bits up to that end split it.
        for (offset, width) in [(0usize, 1usize), (0, 7), (4, 4), (1, 7), (0, 13)] {
            let split_at = if offset % 8 == 0 {
                offset + width
            } else {
                offset
            };
            let lo_size = split_at % 8;
            let hi_max = (1u128 << (8 - lo_size)) - 1;
            let inverse_of_the_low_place = half.modpow(&BigUint::from(lo_size), &modulus);
            for value in [0u128, 5, 200, 255, 0x1fff] {
                let byte = (value >> (split_at / 8 * 8)) & 0xff;
                let honest = byte >> lo_size;
                let accepts = |hi: &BigUint| {
                    accepts_the_forgery::<F>(
                        field,
                        &BigUint::from(value),
                        (offset, width),
                        &Forgery {
                            partial_hi: Some(hi.clone()),
                            ..Forgery::default()
                        },
                    )
                };
                assert!(
                    accepts(&BigUint::from(honest)),
                    "{:?}: bit_range({value}, {offset}, {width}) refuses its honest split",
                    F::ID
                );
                let other_lo = (byte & ((1 << lo_size) - 1)) ^ 1;
                let off_the_integers = (BigUint::from(byte) + &modulus - other_lo)
                    * &inverse_of_the_low_place
                    % &modulus;
                let mut forged: Vec<_> = (honest + 1..=hi_max).take(2).map(BigUint::from).collect();
                forged.extend(honest.checked_sub(1).map(BigUint::from));
                forged.push(off_the_integers);
                for hi in &forged {
                    assert!(
                        !accepts(hi),
                        "{:?}: bit_range({value}, {offset}, {width}) accepts the high part {hi}",
                        F::ID
                    );
                }
            }
        }
    }

    /// Every byte the decomposition sums is held to a byte.
    ///
    /// Moving 256 from a hinted byte to the one below it keeps every sum, so only the lower byte's
    /// check refuses it; where the two straddle the split, a borrow forged to 1 carries the
    /// difference between the halves. The top byte has no byte above it to move from, but written
    /// as the element that makes the high half `-1`, it passes the high distance, which reads that
    /// half modulo `p`, so only its check refuses it. Hinting the bytes of 256 for 5 leaves the
    /// least significant byte `5 - 256`, and every other sum is 5 again, so only its check refuses
    /// it.
    #[test]
    fn every_byte_of_the_decomposition_is_held_to_a_byte() {
        check_every_byte_is_held_to_a_byte::<Field>(FieldConfig::bn254());
        check_every_byte_is_held_to_a_byte::<Goldilocks>(FieldConfig::goldilocks());
    }

    fn check_every_byte_is_held_to_a_byte<F: MavrosField>(field: FieldConfig) {
        let CanonicalBytes { bytes, split } = CanonicalBytes::of(field);
        let modulus = BigUint::from(field.modulus());
        let one = BigUint::from(1u8);
        for place in 1..bytes - 1 {
            let value = &one << (8 * (place + 1));
            let as_elements = |lower: u32, upper: u32| {
                vec![
                    (place, BigUint::from(lower)),
                    (place + 1, BigUint::from(upper)),
                ]
            };
            let honest = Forgery {
                elements: as_elements(0, 1),
                ..Forgery::default()
            };
            assert!(
                accepts_the_forgery::<F>(field, &value, (8, 8), &honest),
                "{:?}: the honest bytes of {value}, written as elements, are refused",
                F::ID
            );
            let forged = Forgery {
                elements: as_elements(256, 0),
                borrow: (8 * (place + 1) == split).then(|| one.clone()),
                ..Forgery::default()
            };
            assert!(
                !accepts_the_forgery::<F>(field, &value, (8, 8), &forged),
                "{:?}: {value} is accepted with a byte of 256 at place {place}",
                F::ID
            );
        }

        let value = &modulus - (&one << split) + 4u8;
        let as_elements = |top: BigUint, below: &dyn Fn(usize) -> BigUint| {
            (1..bytes - 1)
                .map(|place| (place, below(place)))
                .chain([(bytes - 1, top)])
                .collect()
        };
        let byte_of = |place: usize| (&value >> (8 * place)) & BigUint::from(0xffu8);
        let honest = Forgery {
            elements: as_elements(byte_of(bytes - 1), &byte_of),
            ..Forgery::default()
        };
        assert!(
            accepts_the_forgery::<F>(field, &value, (8, 8), &honest),
            "{:?}: the honest bytes of {value}, written as elements, are refused",
            F::ID
        );
        let half: BigUint = (&modulus + 1u8) >> 1u32;
        let top_place_in_the_half = BigUint::from(8 * (bytes - 1) - split);
        let minus_one_as_the_top_byte =
            (&modulus - 1u8) * half.modpow(&top_place_in_the_half, &modulus) % &modulus;
        let forged = Forgery {
            elements: as_elements(minus_one_as_the_top_byte, &|_| BigUint::ZERO),
            ..Forgery::default()
        };
        assert!(
            !accepts_the_forgery::<F>(field, &value, (8, 8), &forged),
            "{:?}: {value} is accepted with a high half of -1",
            F::ID
        );

        let five = BigUint::from(5u8);
        let bytes_of = |representative: u16| Forgery {
            bytes: Some(BigUint::from(representative)),
            ..Forgery::default()
        };
        assert!(
            accepts_the_forgery::<F>(field, &five, (0, 8), &bytes_of(5)),
            "{:?}: the honest bytes of 5 are refused",
            F::ID
        );
        assert!(
            !accepts_the_forgery::<F>(field, &five, (0, 8), &bytes_of(256)),
            "{:?}: 5 is accepted with the bytes of 256",
            F::ID
        );
    }

    /// The hints a dishonest prover replaces, each [`None`] or empty where it stays honest.
    #[derive(Debug, Default)]
    struct Forgery {
        /// The decomposition's bytes, as those of this integer.
        bytes: Option<BigUint>,
        /// Hinted bytes written as these elements, each named by its place counted up from the
        /// least significant byte, which is not hinted.
        elements: Vec<(usize, BigUint)>,
        /// The borrow of the comparison against `p - 1`, as this element.
        borrow: Option<BigUint>,
        /// The high part of the one byte a window splits, as this element.
        partial_hi: Option<BigUint>,
    }

    /// Whether `main(x: Field) -> Field { bit_range(x, offset, width) }` accepts `value` with the
    /// hints `forgery` names replaced.
    fn accepts_the_forgery<F: MavrosField>(
        field: FieldConfig,
        value: &BigUint,
        (offset, width): (usize, usize),
        forgery: &Forgery,
    ) -> bool {
        let count = CanonicalBytes::of(field).bytes;
        let mut ssa = HLSSA::with_main_in(field, "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b
                .block(entry)
                .with_source_location(SourceLocation::synthetic("canonical_bytes"));
            let x = e.add_parameter(Type::field());
            let window = e.bit_range(x, offset, width);
            e.terminate_return(vec![window]);
        });

        let element = |value: &BigUint| {
            field
                .from_bigint(value.clone().try_into().unwrap())
                .expect("a canonical value")
        };
        let input = InputValueOrdered::Field(F::from_element(element(value)));
        forgery::accepts_the_forgery::<F>(ssa, &[input], |ssa| {
            for (place, forged) in &forgery.elements {
                assert!(*place > 0, "the least significant byte is not hinted");
                let forged = ssa.add_const(Constant::Field(element(forged)));
                let index = ssa.find_const(&Constant::Int(IntBits::from_u128(
                    32,
                    (count - 1 - place) as u128,
                )));
                let radix: Vec<_> = all_ops(ssa)
                    .filter_map(|op| match op {
                        OpCode::ToRadix { result, .. } => Some(result),
                        _ => None,
                    })
                    .collect();
                let reads: Vec<_> = all_ops(ssa)
                    .filter_map(|op| match op {
                        OpCode::ArrayGet {
                            result,
                            array,
                            index: at,
                        } if radix.contains(&array) && Some(at) == index => Some(result),
                        _ => None,
                    })
                    .collect();
                let hints: Vec<_> = all_ops(ssa)
                    .filter_map(|op| match op {
                        OpCode::Cast {
                            result,
                            value,
                            target: CastTarget::Field,
                        } if reads.contains(&value) => Some(result),
                        _ => None,
                    })
                    .collect();
                let replaced = replace_hints(ssa, |op| match *op {
                    OpCode::WriteWitness {
                        result,
                        value,
                        pinned,
                    } if hints.contains(&value) => Some(OpCode::WriteWitness {
                        result,
                        value: forged,
                        pinned,
                    }),
                    _ => None,
                });
                assert_eq!(replaced, 1, "byte {place} is one hint");
            }
            if let Some(representative) = &forgery.bytes {
                let mut bytes = representative.to_bytes_be();
                bytes.splice(0..0, std::iter::repeat_n(0, count - bytes.len()));
                let elems: Vec<_> = bytes
                    .into_iter()
                    .map(|byte| ssa.add_const(Constant::Int(IntBits::from_u128(8, byte.into()))))
                    .collect();
                let replaced = replace_hints(ssa, |op| match *op {
                    OpCode::ToRadix { result, .. } => Some(OpCode::MkSeq {
                        result,
                        elems: elems.clone(),
                        seq_type: SequenceTargetType::Array(count),
                        elem_type: Type::int(8),
                    }),
                    _ => None,
                });
                assert_eq!(
                    replaced, 1,
                    "the decomposition is the program's one byte hint"
                );
            }
            if let Some(borrow) = &forgery.borrow {
                // Found by either of its uses, its boolean constraint or its shift to the split, so
                // that deleting one of the checks it feeds leaves the other to find it.
                let forged = ssa.add_const(Constant::Field(element(borrow)));
                let split = CanonicalBytes::of(field).split;
                let shift = ssa.find_const(&Constant::Field(field.two_pow(split)));
                let uses: Vec<_> = all_ops(ssa)
                    .filter_map(|op| match op {
                        OpCode::Constrain { a, b, c } if a == b && b == c => Some(a),
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::UMul,
                            lhs,
                            rhs,
                            ..
                        } if Some(rhs) == shift => Some(lhs),
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::UMul,
                            lhs,
                            rhs,
                            ..
                        } if Some(lhs) == shift => Some(rhs),
                        _ => None,
                    })
                    .collect();
                let replaced = replace_hints(ssa, |op| match *op {
                    OpCode::WriteWitness {
                        result: Some(result),
                        pinned,
                        ..
                    } if uses.contains(&result) => Some(OpCode::WriteWitness {
                        result: Some(result),
                        value: forged,
                        pinned,
                    }),
                    _ => None,
                });
                assert_eq!(replaced, 1, "the borrow is the program's one such witness");
            }
            if let Some(hi) = &forgery.partial_hi {
                let forged = ssa.add_const(Constant::Field(element(hi)));
                let quotients: Vec<_> = all_ops(ssa)
                    .filter_map(|op| match op {
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::UDiv,
                            result,
                            ..
                        } => Some(result),
                        _ => None,
                    })
                    .collect();
                let hints: Vec<_> = all_ops(ssa)
                    .filter_map(|op| match op {
                        OpCode::Cast {
                            result,
                            value,
                            target: CastTarget::Field,
                        } if quotients.contains(&value) => Some(result),
                        _ => None,
                    })
                    .collect();
                let replaced = replace_hints(ssa, |op| match *op {
                    OpCode::WriteWitness {
                        result,
                        value,
                        pinned,
                    } if hints.contains(&value) => Some(OpCode::WriteWitness {
                        result,
                        value: forged,
                        pinned,
                    }),
                    _ => None,
                });
                assert_eq!(replaced, 1, "the split is the program's one quotient hint");
            }
        })
    }
}
