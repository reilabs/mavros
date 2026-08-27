use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use mavros_artifacts::FieldConfig;

use crate::compiler::{
    analysis::value_range_analysis::{Interval, field_modulus},
    passes::instruction_lowering::{InstructionLoweringRule, LoweringContext, integer_bits},
    ssa::{
        ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CastTarget, CmpKind, OpCode, assert_signed_op_width,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

pub struct LowerWitnessIntegerArithOps {}

impl InstructionLoweringRule for LowerWitnessIntegerArithOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (guard, op) = HLBlockEmitter::unwrap_guard(instruction);
        self.process_arith(b, context, guard, op)
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
                kind,
                result,
                lhs,
                rhs,
            } if self.should_lower_integer_arith(context, *lhs, *rhs) => {
                // The bitwise operations and the shifts are `LowerWitnessBitwiseOps`' business, and
                // this rule runs first in `witness_integer_ops`, so it has to decline them for
                // that one to see them. Returning `false` is what passes them on; the `unreachable`
                // at the bottom of the match is the other half of the same statement.
                if !matches!(
                    kind.group(),
                    ArithGroup::Add
                        | ArithGroup::Sub
                        | ArithGroup::Mul
                        | ArithGroup::Div
                        | ArithGroup::Rem
                ) {
                    return false;
                }

                let lhs_ty = context.types().get_value_type(*lhs);
                let bits = integer_bits(lhs_ty).unwrap();
                // The operation picks the lowering; the type above supplied only the width. Every
                // arm below is handed `kind` itself rather than a rebuilt one, so the hints and
                // helper operations a lowering emits carry the same sign as the operation that
                // selected it.
                let signed = kind.is_signed();
                let kind = *kind;

                match kind.group() {
                    ArithGroup::Add | ArithGroup::Sub => {
                        if signed {
                            self.lower_signed_addsub(
                                b, context, guard, kind, *result, *lhs, *rhs, bits,
                            );
                        } else {
                            self.lower_unsigned_addsub(
                                b, context, guard, kind, *result, *lhs, *rhs, bits,
                            );
                        }
                        true
                    }
                    ArithGroup::Mul => {
                        if signed {
                            self.lower_signed_mul(b, context, guard, *result, *lhs, *rhs, bits);
                        } else {
                            self.lower_unsigned_mul(b, context, guard, *result, *lhs, *rhs, bits);
                        }
                        true
                    }
                    ArithGroup::Div | ArithGroup::Rem => {
                        if signed {
                            self.lower_signed_divmod(
                                b, context, guard, kind, *result, *lhs, *rhs, bits,
                            );
                        } else {
                            self.lower_unsigned_divmod_result(
                                b, context, guard, kind, *result, *lhs, *rhs, bits,
                            );
                        }
                        true
                    }
                    // Rejected above, before the sign was resolved.
                    ArithGroup::And
                    | ArithGroup::Or
                    | ArithGroup::Xor
                    | ArithGroup::Shl
                    | ArithGroup::Shr => unreachable!("filtered out above"),
                }
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
        (lhs_ty.is_witness_of() || rhs_ty.is_witness_of()) && integer_bits(lhs_ty).is_some()
    }

    // FIELD-ASSUMPTION: L6-int-op-strategy
    // The sum/difference is computed in one field element. Sound while `2^(bits+1) < p`; a
    // u64 sum (65 bits) overflows a ~64-bit field and needs a carry-chain lowering.
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
        // The sum is a _field_ value, so the unsigned forms are the right ones to build it with.
        let value = match kind.group() {
            ArithGroup::Add => b.uadd(lhs_field, rhs_field),
            ArithGroup::Sub => b.usub(lhs_field, rhs_field),
            _ => unreachable!(),
        };
        let range = match kind.group() {
            ArithGroup::Add => context.urange(lhs).add(&context.urange(rhs)),
            ArithGroup::Sub => context.urange(lhs).sub(&context.urange(rhs)),
            _ => unreachable!(),
        };
        if !range.proves_fits_in_unsigned_bits(bits) {
            let rc_bits = narrow_rangecheck_width(&range, bits);
            guarded_rangecheck(b, value, rc_bits, guard);
        }
        let value = guarded_or_zero_field(b, value, guard);
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
        });
    }

    // FIELD-ASSUMPTION: L6-int-op-strategy
    // The full product is computed in one field element (guarded by
    // `range_fits_field_injectively`). The only non-single-field path is the u128 fallback
    // below, and it still packs `lo + cross*2^64` (~2^193) into one cell — so it assumes ~193
    // bits of headroom. On a small field u32/u64 mul need a schoolbook multi-limb lowering
    // with per-limb range checks and carries (see docs/field-agnosticism.md, Layer 6).
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
        let product_range = context.urange(lhs).mul(&context.urange(rhs));
        if bits == 128 && !range_fits_field_injectively(&product_range, b.field()) {
            let lhs_limbs = split_u128_value(b, lhs);
            let rhs_limbs = split_u128_value(b, rhs);
            let lhs_lo = b.cast_to_field(lhs_limbs.lo);
            let lhs_hi = b.cast_to_field(lhs_limbs.hi);
            let rhs_lo = b.cast_to_field(rhs_limbs.lo);
            let rhs_hi = b.cast_to_field(rhs_limbs.hi);

            let lo_product = b.umul(lhs_lo, rhs_lo);
            let lhs_cross = b.umul(lhs_lo, rhs_hi);
            let rhs_cross = b.umul(lhs_hi, rhs_lo);
            let high_product = b.umul(lhs_hi, rhs_hi);
            let zero = b.field_const(b.field().zero());
            let flag = guard
                .map(|condition| b.cast_to_field(condition))
                .unwrap_or_else(|| b.field_const(b.field().one()));
            b.constrain(flag, high_product, zero);

            let cross = b.uadd(lhs_cross, rhs_cross);
            let shift = b.field_const(b.field().two_pow(64));
            let shifted_cross = b.umul(cross, shift);
            let value = b.uadd(lo_product, shifted_cross);
            guarded_rangecheck(b, value, bits, guard);
            let value = guarded_or_zero_field(b, value, guard);
            b.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::Int(bits),
            });
            return;
        }

        assert!(
            range_fits_field_injectively(&product_range, b.field()),
            "unsigned multiplication product range is too wide for a single-field product"
        );

        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let value = b.umul(lhs_field, rhs_field);
        if !product_range.proves_fits_in_unsigned_bits(bits) {
            let rc_bits = narrow_rangecheck_width(&product_range, bits);
            guarded_rangecheck(b, value, rc_bits, guard);
        }
        let value = guarded_or_zero_field(b, value, guard);
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
        });
    }

    #[allow(clippy::too_many_arguments)]
    // FIELD-ASSUMPTION: L6-int-op-strategy
    // Operands are decoded to signed field values and added/subtracted in one field element,
    // asserting the result range fits injectively. i64 breaks on a ~64-bit field because the
    // `sign * 2^bits` re-encoding offset alone exceeds p (see `encode_signed_value`).
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
        assert_signed_op_width(bits, "signed addition");
        let lhs_range = context.srange(lhs);
        let rhs_range = context.srange(rhs);
        let sign_l = match known_sign(&lhs_range, bits) {
            Some(false) => b.field_const(b.field().zero()),
            Some(true) => b.field_const(b.field().one()),
            None => {
                let sign_l_bits = b.bit_range(lhs, bits - 1, 1);
                b.cast_to_field(sign_l_bits)
            }
        };
        let sign_r = match known_sign(&rhs_range, bits) {
            Some(false) => b.field_const(b.field().zero()),
            Some(true) => b.field_const(b.field().one()),
            None => {
                let sign_r_bits = b.bit_range(rhs, bits - 1, 1);
                b.cast_to_field(sign_r_bits)
            }
        };
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let lhs_signed = signed_value_from_encoded(b, lhs_field, sign_l, bits);
        let rhs_signed = signed_value_from_encoded(b, rhs_field, sign_r, bits);

        let result_range = match kind.group() {
            ArithGroup::Add => lhs_range.add(&rhs_range),
            ArithGroup::Sub => lhs_range.sub(&rhs_range),
            _ => unreachable!(),
        };
        assert!(
            range_fits_field_injectively(&result_range, b.field()),
            "signed add/sub result range is too wide for a single-field encoding"
        );

        // `lhs_signed`/`rhs_signed` are decoded _field_ values, so this is field arithmetic and
        // takes the unsigned forms even though the operands it came from are signed integers.
        let signed_raw = match kind.group() {
            ArithGroup::Add => b.uadd(lhs_signed, rhs_signed),
            ArithGroup::Sub => b.usub(lhs_signed, rhs_signed),
            _ => unreachable!(),
        };

        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        // Unlike `signed_raw` above, this one is computed on the _integer_ operands at their own
        // width, so it carries the original operation — including its sign — rather than an
        // unsigned stand-in.
        let result_hint = b.bin(kind, lhs_pure, rhs_pure);
        let result_hint_unsigned = b.cast_to(CastTarget::Int(bits), result_hint);
        let result_value = encode_signed_value(
            b,
            signed_raw,
            result_hint_unsigned,
            &result_range,
            bits,
            guard,
        );
        b.emit(OpCode::Cast {
            result,
            value: result_value,
            target: CastTarget::Int(bits),
        });
    }

    // FIELD-ASSUMPTION: L6-int-op-strategy
    // Single field mul with no multi-limb fallback at all (unlike unsigned u128). Sound only
    // while the signed product range fits the field; i32/i64 mul need a schoolbook lowering
    // on a small field.
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
        assert_signed_op_width(bits, "signed multiplication");
        let lhs_range = context.srange(lhs);
        let rhs_range = context.srange(rhs);
        let product_range = lhs_range.mul(&rhs_range);
        assert!(
            range_fits_field_injectively(&product_range, b.field()),
            "signed multiplication product range is too wide for a single-field product"
        );

        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let sign_l = match known_sign(&lhs_range, bits) {
            Some(false) => b.field_const(b.field().zero()),
            Some(true) => b.field_const(b.field().one()),
            None => {
                let sign_l_bits = b.bit_range(lhs, bits - 1, 1);
                b.cast_to_field(sign_l_bits)
            }
        };
        let sign_r = match known_sign(&rhs_range, bits) {
            Some(false) => b.field_const(b.field().zero()),
            Some(true) => b.field_const(b.field().one()),
            None => {
                let sign_r_bits = b.bit_range(rhs, bits - 1, 1);
                b.cast_to_field(sign_r_bits)
            }
        };
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);
        let lhs_signed = signed_value_from_encoded(b, lhs_field, sign_l, bits);
        let rhs_signed = signed_value_from_encoded(b, rhs_field, sign_r, bits);

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        // The hint is a _signed_ multiply — `smul` is what says so, the operand type no longer
        // can — taken at the operands' own width. It is the sibling of the `result_hint` in
        // `lower_signed_addsub`. The product below is on decoded field values and stays unsigned.
        let result_hint = b.smul(lhs_pure, rhs_pure);
        let result_hint_unsigned = b.cast_to(CastTarget::Int(bits), result_hint);
        let product = b.umul(lhs_signed, rhs_signed);
        let result_value = encode_signed_value(
            b,
            product,
            result_hint_unsigned,
            &product_range,
            bits,
            guard,
        );

        b.emit(OpCode::Cast {
            result,
            value: result_value,
            target: CastTarget::Int(bits),
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
        let divmod = lower_unsigned_divmod(
            b,
            lhs,
            rhs,
            bits,
            context.types().get_value_type(lhs).is_witness_of(),
            context.types().get_value_type(rhs).is_witness_of(),
            &context.urange(lhs),
            &context.urange(rhs),
            guard,
            guard_is_witness,
        );
        let value = match kind.group() {
            ArithGroup::Div => divmod.q,
            ArithGroup::Rem => divmod.r,
            _ => unreachable!(),
        };
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
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
        assert_signed_op_width(bits, "signed division");
        let lhs_witness = context.types().get_value_type(lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        let lhs_range = context.srange(lhs);
        let rhs_range = context.srange(rhs);

        let lhs_known_sign = known_sign(&lhs_range, bits);
        let sign_l_is_witness = lhs_witness && lhs_known_sign.is_none();
        let (sign_l_u1, sign_l) = match lhs_known_sign {
            Some(false) => (b.int_const(1, 0), b.field_const(b.field().zero())),
            Some(true) => (b.int_const(1, 1), b.field_const(b.field().one())),
            None => {
                let sign_l_bits = b.bit_range(lhs, bits - 1, 1);
                let sign_l_u1 = b.cast_to(CastTarget::Int(1), sign_l_bits);
                let sign_l = b.cast_to_field(sign_l_u1);
                (sign_l_u1, sign_l)
            }
        };
        let rhs_known_sign = known_sign(&rhs_range, bits);
        let sign_r_is_witness = if lhs == rhs {
            sign_l_is_witness
        } else {
            rhs_witness && rhs_known_sign.is_none()
        };
        let (sign_r_u1, sign_r) = if lhs == rhs {
            (sign_l_u1, sign_l)
        } else {
            match rhs_known_sign {
                Some(false) => (b.int_const(1, 0), b.field_const(b.field().zero())),
                Some(true) => (b.int_const(1, 1), b.field_const(b.field().one())),
                None => {
                    let sign_r_bits = b.bit_range(rhs, bits - 1, 1);
                    let sign_r_u1 = b.cast_to(CastTarget::Int(1), sign_r_bits);
                    let sign_r = b.cast_to_field(sign_r_u1);
                    (sign_r_u1, sign_r)
                }
            }
        };
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);

        let abs_l = self.write_abs_value(b, lhs_field, sign_l, bits);
        let abs_r = if lhs == rhs {
            abs_l
        } else {
            self.write_abs_value(b, rhs_field, sign_r, bits)
        };

        let abs_l_range = abs_bound(&lhs_range);
        let abs_r_range = if lhs == rhs {
            abs_l_range.clone()
        } else {
            abs_bound(&rhs_range)
        };
        let abs_l_is_witness = lhs_witness || sign_l_is_witness;
        let abs_r_is_witness = if lhs == rhs {
            abs_l_is_witness
        } else {
            rhs_witness || sign_r_is_witness
        };
        let guard_is_witness = guard
            .map(|condition| context.types().get_value_type(condition).is_witness_of())
            .unwrap_or(false);
        let divmod = lower_unsigned_divmod(
            b,
            abs_l,
            abs_r,
            bits,
            abs_l_is_witness,
            abs_r_is_witness,
            &abs_l_range,
            &abs_r_range,
            guard,
            guard_is_witness,
        );

        let quotient_sign_u1 = b.xor(sign_l_u1, sign_r_u1);
        let quotient_sign = b.cast_to_field(quotient_sign_u1);
        let quotient_sign_is_witness = sign_l_is_witness || sign_r_is_witness;

        let quotient = self.write_signed_magnitude_result(
            b,
            divmod.q,
            divmod.q_is_witness,
            quotient_sign,
            quotient_sign_u1,
            quotient_sign_is_witness,
            bits,
            guard,
        );
        let remainder = self.write_signed_magnitude_result(
            b,
            divmod.r,
            divmod.r_is_witness,
            sign_l,
            sign_l_u1,
            sign_l_is_witness,
            bits,
            guard,
        );

        let value = match kind.group() {
            ArithGroup::Div => quotient,
            ArithGroup::Rem => remainder,
            _ => unreachable!(),
        };
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
        });
    }

    fn write_abs_value(
        &self,
        b: &mut HLBlockEmitter<'_>,
        value_field: ValueId,
        sign: ValueId,
        bits: usize,
    ) -> ValueId {
        let signed_value = signed_value_from_encoded(b, value_field, sign, bits);
        let two = b.field_const(b.field().constant(2));
        let two_sign = b.umul(two, sign);
        let one = b.field_const(b.field().one());
        let factor = b.usub(one, two_sign);
        b.umul(signed_value, factor)
    }

    fn write_signed_magnitude_result(
        &self,
        b: &mut HLBlockEmitter<'_>,
        magnitude: ValueId,
        magnitude_is_witness: bool,
        sign: ValueId,
        sign_u1: ValueId,
        sign_is_witness: bool,
        bits: usize,
        guard: Option<ValueId>,
    ) -> ValueId {
        let magnitude_pure = if magnitude_is_witness {
            b.value_of(magnitude)
        } else {
            magnitude
        };
        let sign_for_hint = if sign_is_witness {
            b.value_of(sign_u1)
        } else {
            sign_u1
        };
        let magnitude_field = b.cast_to_field(magnitude_pure);
        let two_n_field = b.field_const(b.field().two_pow(bits));
        let neg = b.usub(two_n_field, magnitude_field);
        let encoded_if_nonzero = b.select(sign_for_hint, neg, magnitude_field);
        let zero = b.field_const(b.field().zero());
        let magnitude_is_zero = b.eq(magnitude_field, zero);
        let two = b.field_const(b.field().constant(2));
        let two_sign = b.umul(two, sign);
        let one = b.field_const(b.field().one());
        let factor = b.usub(one, two_sign);
        let signed_value = b.umul(magnitude, factor);

        let encoded_hint = b.select(magnitude_is_zero, zero, encoded_if_nonzero);
        let encoded_hint = b.cast_to(CastTarget::Int(bits), encoded_hint);
        encode_signed_value(b, signed_value, encoded_hint, &Interval::top(), bits, guard)
    }
}

fn guarded_rangecheck(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    bits: usize,
    guard: Option<ValueId>,
) {
    assert!(bits >= 1, "rangecheck width must be at least 1 bit");
    b.emit_guarded(
        guard,
        OpCode::Rangecheck {
            value,
            max_bits: bits,
        },
    );
}

pub(super) fn guarded_or_zero_field(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    guard: Option<ValueId>,
) -> ValueId {
    if let Some(condition) = guard {
        let zero = b.field_const(b.field().zero());
        b.select(condition, value, zero)
    } else {
        value
    }
}

fn narrow_rangecheck_width(range: &Interval, default_bits: usize) -> usize {
    // ⊥ is `[1, 0]`, which would otherwise measure as a _one-bit_ check on a value the analysis
    // only believes unreachable because of this very check. Fall back to the declared width.
    if range.is_empty() {
        return default_bits;
    }
    let (Some(lo), Some(hi)) = (range.lo(), range.hi()) else {
        return default_bits;
    };
    if lo.is_negative() {
        return default_bits;
    }
    let width = hi.bits() as usize;
    width.max(1).min(default_bits)
}

// FIELD-ASSUMPTION: L4-modulus-query
// This gate already reads the real modulus and flips correctly on a small field (it returns
// false for u64xu64, and even for the fragile u32xu32 edge). The debt is not the predicate —
// it is the missing FALSE-case codegen (only unsigned u128 mul has a fallback). See Layer 6
// (`L6-int-op-strategy`) in docs/field-agnosticism.md.
fn range_fits_field_injectively(range: &Interval, field: FieldConfig) -> bool {
    // ⊥ would report `hi - lo = -1 < p` and so license the single-field product path on no evidence
    // at all.
    if range.is_empty() {
        return false;
    }
    let Some(lo) = range.lo() else {
        return false;
    };
    let Some(hi) = range.hi() else {
        return false;
    };
    let p = field_modulus(field);
    // All integer representatives in this range have distinct field encodings
    // when their pairwise distance is less than p.
    hi - lo < p
}

// FIELD-ASSUMPTION: L6-int-op-strategy (signed encode/decode pair)
// `signed_value_from_encoded`/`encode_signed_value` pack the sign with `field.two_pow(bits)` /
// `field.two_pow(bits-1)` place-value shifts. These packings wrap mod p once the shift reaches the
// field width, so i64 sign encoding is unsound on a small field.
fn signed_value_from_encoded(
    b: &mut HLBlockEmitter<'_>,
    encoded_field: ValueId,
    sign: ValueId,
    bits: usize,
) -> ValueId {
    let sign_shift = b.field_const(b.field().two_pow(bits));
    let sign_shifted = b.umul(sign, sign_shift);
    b.usub(encoded_field, sign_shifted)
}

/// The provable sign of a value, or `None` when it is not known. Callers hardcode a constant sign
/// bit on the strength of this, so ⊥ must answer `None` rather than the vacuous `Some(false)`.
fn known_sign(range: &Interval, bits: usize) -> Option<bool> {
    if range.proves_non_negative_in_signed(bits) {
        Some(false)
    } else if is_strictly_negative(range) {
        Some(true)
    } else {
        None
    }
}

fn encode_signed_value(
    b: &mut HLBlockEmitter<'_>,
    signed_value: ValueId,
    encoded_hint: ValueId,
    signed_range: &Interval,
    bits: usize,
    guard: Option<ValueId>,
) -> ValueId {
    let sign = if signed_range.proves_non_negative_in_signed(bits) {
        b.field_const(b.field().zero())
    } else if is_strictly_negative(signed_range) {
        b.field_const(b.field().one())
    } else {
        let sign_hint = b.bit_range(encoded_hint, bits - 1, 1);
        let sign_hint = b.cast_to_field(sign_hint);
        let sign = b.write_witness(sign_hint);
        guarded_rangecheck(b, sign, 1, guard);
        sign
    };

    let sign_shift = b.field_const(b.field().two_pow(bits));
    let sign_shifted = b.umul(sign, sign_shift);
    let encoded = b.uadd(signed_value, sign_shifted);
    if !signed_range.proves_fits_in_signed_bits(bits) || known_sign(signed_range, bits).is_none() {
        if bits == 1 {
            let diff = b.usub(encoded, sign);
            guarded_rangecheck(b, diff, 1, guard);
            let neg_diff = b.usub(sign, encoded);
            guarded_rangecheck(b, neg_diff, 1, guard);
        } else {
            let half = b.field_const(b.field().two_pow(bits - 1));
            let sign_half = b.umul(sign, half);
            let sign_limb = b.usub(encoded, sign_half);
            guarded_rangecheck(b, sign_limb, bits - 1, guard);
        }
    }
    guarded_or_zero_field(b, encoded, guard)
}

fn is_strictly_negative(range: &Interval) -> bool {
    range.hi().is_some_and(|hi| hi.is_negative())
}

struct DivModResult {
    q: ValueId,
    r: ValueId,
    q_is_witness: bool,
    r_is_witness: bool,
}

#[derive(Clone, Copy)]
struct U128Limbs {
    lo: ValueId,
    hi: ValueId,
}

// FIELD-ASSUMPTION: L6-int-representation
// A fixed 2x64-bit split of a u128. On a small field the limb width must derive from the
// field size (h ~= field_bits/2), and wide integers become multi-cell values end-to-end.
fn split_u128_value(b: &mut impl HLEmitter, value: ValueId) -> U128Limbs {
    let value = b.cast_to(CastTarget::Int(128), value);
    U128Limbs {
        lo: split_u128_limb(b, value, 0),
        hi: split_u128_limb(b, value, 64),
    }
}

fn split_u128_limb(b: &mut impl HLEmitter, value: ValueId, offset: usize) -> ValueId {
    let limb = b.bit_range(value, offset, 64);
    b.cast_to(CastTarget::Int(64), limb)
}

// FIELD-ASSUMPTION: L6-int-op-strategy
// Reconstructs `dividend = q * divisor + r` in one field element; the u128 path recurses into
// the u128 mul fallback. The reconstruction overflows a small field, and even the fragile
// u32/u64 `+` fused into `q*divisor + r` can tip past p (needs the multi-limb mul engine).
#[allow(clippy::too_many_arguments)]
fn lower_unsigned_divmod(
    b: &mut HLBlockEmitter<'_>,
    dividend: ValueId,
    divisor: ValueId,
    bits: usize,
    dividend_is_witness: bool,
    divisor_is_witness: bool,
    dividend_range: &Interval,
    divisor_range: &Interval,
    guard: Option<ValueId>,
    guard_is_witness: bool,
) -> DivModResult {
    if bits == 128 {
        if dividend == divisor {
            let active = if let Some(condition) = guard {
                b.cast_to_field(condition)
            } else {
                b.field_const(b.field().one())
            };
            let zero = b.field_const(b.field().zero());
            let one = b.field_const(b.field().one());
            let divisor_field = b.cast_to_field(divisor);
            let divisor_minus_one = b.usub(divisor_field, one);
            guarded_rangecheck(b, divisor_minus_one, 128, guard);
            return DivModResult {
                q: active,
                r: zero,
                q_is_witness: guard_is_witness,
                r_is_witness: false,
            };
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

        let mut dividend_hint = b.cast_to(CastTarget::Int(128), dividend_pure);
        let mut divisor_hint = b.cast_to(CastTarget::Int(128), divisor_pure);
        if let Some(condition) = guard {
            let condition = if guard_is_witness {
                b.value_of(condition)
            } else {
                condition
            };
            let zero = b.int_const(128, 0);
            let one = b.int_const(128, 1);
            dividend_hint = b.select(condition, dividend_hint, zero);
            divisor_hint = b.select(condition, divisor_hint, one);
        }

        let q_hint = b.udiv(dividend_hint, divisor_hint);
        let r_hint = b.urem(dividend_hint, divisor_hint);
        let q_hint_field = b.cast_to_field(q_hint);
        let r_hint_field = b.cast_to_field(r_hint);
        let q_wit = b.write_witness(q_hint_field);
        let r_wit = b.write_witness(r_hint_field);
        guarded_rangecheck(
            b,
            q_wit,
            narrow_rangecheck_width(&quotient_bound(dividend_range, divisor_range), 128),
            guard,
        );
        guarded_rangecheck(
            b,
            r_wit,
            narrow_rangecheck_width(&remainder_bound(divisor_range), 128),
            guard,
        );

        let r_u128 = b.cast_to(CastTarget::Int(128), r_wit);
        let q_u128 = b.cast_to(CastTarget::Int(128), q_wit);
        let product = b.fresh_value();
        b.emit_guarded(
            guard,
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UMul,
                result: product,
                lhs: divisor,
                rhs: q_u128,
            },
        );
        let sum = b.fresh_value();
        b.emit_guarded(
            guard,
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UAdd,
                result: sum,
                lhs: product,
                rhs: r_u128,
            },
        );
        b.emit_guarded(
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: sum,
                rhs: dividend,
            },
        );
        b.emit_guarded(
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::ULt,
                lhs: r_u128,
                rhs: divisor,
            },
        );

        return DivModResult {
            q: guarded_or_zero_field(b, q_wit, guard),
            r: guarded_or_zero_field(b, r_wit, guard),
            q_is_witness: true,
            r_is_witness: true,
        };
    }

    if dividend == divisor {
        let active = if let Some(condition) = guard {
            b.cast_to_field(condition)
        } else {
            b.field_const(b.field().one())
        };
        let zero = b.field_const(b.field().zero());
        let one = b.field_const(b.field().one());

        let divisor_field = b.cast_to_field(divisor);
        let divisor_minus_one = b.usub(divisor_field, one);
        guarded_rangecheck(b, divisor_minus_one, bits, guard);

        return DivModResult {
            q: active,
            r: zero,
            q_is_witness: guard_is_witness,
            r_is_witness: false,
        };
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

    let mut dividend_hint = b.cast_to(CastTarget::Int(bits), dividend_pure);
    let mut divisor_hint = b.cast_to(CastTarget::Int(bits), divisor_pure);
    if let Some(condition) = guard {
        let condition = if guard_is_witness {
            b.value_of(condition)
        } else {
            condition
        };
        let zero = b.int_const(bits, 0);
        let one = b.int_const(bits, 1);
        dividend_hint = b.select(condition, dividend_hint, zero);
        divisor_hint = b.select(condition, divisor_hint, one);
    }
    let q_hint = b.udiv(dividend_hint, divisor_hint);
    let q_hint_field = b.cast_to_field(q_hint);
    let q_wit = b.write_witness(q_hint_field);

    let dividend_field = b.cast_to_field(dividend);
    let divisor_field = b.cast_to_field(divisor);
    let product = b.umul(q_wit, divisor_field);
    let r_raw = b.usub(dividend_field, product);

    let q_bits = narrow_rangecheck_width(&quotient_bound(dividend_range, divisor_range), bits);
    let r_bound = remainder_bound(divisor_range);
    let r_bits = narrow_rangecheck_width(&r_bound, bits);
    guarded_rangecheck(b, q_wit, q_bits, guard);
    guarded_rangecheck(b, r_raw, r_bits, guard);

    let one = b.field_const(b.field().one());
    let divisor_minus_r = b.usub(divisor_field, r_raw);
    let divisor_minus_r_minus_one = b.usub(divisor_minus_r, one);
    guarded_rangecheck(b, divisor_minus_r_minus_one, r_bits, guard);

    DivModResult {
        q: guarded_or_zero_field(b, q_wit, guard),
        r: guarded_or_zero_field(b, r_raw, guard),
        q_is_witness: true,
        r_is_witness: true,
    }
}

fn quotient_bound(a_range: &Interval, b_range: &Interval) -> Interval {
    let (Some(a_hi), Some(b_lo)) = (a_range.hi(), b_range.lo()) else {
        return Interval::top();
    };
    if !a_range.is_non_negative() || !b_lo.is_positive() {
        return Interval::top();
    }
    Interval::closed(BigInt::zero(), a_hi / b_lo)
}

fn remainder_bound(b_range: &Interval) -> Interval {
    let Some(b_hi) = b_range.hi() else {
        return Interval::top();
    };
    if !b_hi.is_positive() {
        return Interval::top();
    }
    Interval::closed(BigInt::zero(), b_hi - BigInt::one())
}

fn abs_bound(range: &Interval) -> Interval {
    let Some(lo) = range.lo() else {
        return Interval::top();
    };
    let Some(hi) = range.hi() else {
        return Interval::top();
    };
    let lo_abs = lo.abs();
    let hi_abs = hi.abs();
    let max = if lo_abs >= hi_abs {
        lo_abs.clone()
    } else {
        hi_abs.clone()
    };
    if lo <= &BigInt::zero() && hi >= &BigInt::zero() {
        Interval::closed(BigInt::zero(), max)
    } else {
        let min = if lo_abs <= hi_abs { lo_abs } else { hi_abs };
        Interval::closed(min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ⊥ means "the range analysis believes this value cannot occur" — but the belief rests on the
    // constraints this module is about to emit, so none of these helpers may treat it as evidence.
    // Left unguarded, `narrow_rangecheck_width` measures `[1, 0]` as a _one-bit_ check.

    #[test]
    fn bottom_does_not_narrow_a_rangecheck() {
        assert_eq!(narrow_rangecheck_width(&Interval::empty(), 32), 32);
        // The surrounding behavior is unchanged: a genuine bound still narrows, and an unbounded
        // or possibly-negative one still falls back.
        assert_eq!(narrow_rangecheck_width(&Interval::closed(0, 100), 32), 7);
        assert_eq!(narrow_rangecheck_width(&Interval::closed(-1, 100), 32), 32);
        assert_eq!(narrow_rangecheck_width(&Interval::top(), 32), 32);
    }

    #[test]
    fn bottom_has_no_known_sign() {
        // Without the guard this is `Some(false)`, which hardcodes `sign = 0` into the encoding.
        assert_eq!(known_sign(&Interval::empty(), 8), None);
        assert_eq!(known_sign(&Interval::closed(0, 4), 8), Some(false));
        assert_eq!(known_sign(&Interval::closed(-4, -1), 8), Some(true));
        assert_eq!(known_sign(&Interval::closed(-4, 4), 8), None);
    }

    #[test]
    fn bottom_is_not_field_injective() {
        // `hi - lo` is `-1` for ⊥, which is trivially below any modulus.
        assert!(!range_fits_field_injectively(
            &Interval::empty(),
            FieldConfig::bn254()
        ));
        assert!(range_fits_field_injectively(
            &Interval::closed(0, 1000),
            FieldConfig::bn254()
        ));
        assert!(!range_fits_field_injectively(
            &Interval::top(),
            FieldConfig::bn254()
        ));
    }
}
