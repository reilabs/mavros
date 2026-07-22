//! Lowers integer bitwise, bit-selection, and sign-extension operations before the main
//! explicit-witness pass.
//!
//! This pass emits `Spread`/`Unspread` operations, except for `u64` bitwise ops where it keeps a
//! two-limb `u32` decomposition. It also canonicalizes witness integer casts/shifts into the shared
//! `BitRange` representation where possible.

use ark_ff::{AdditiveGroup as _, Field as _};

use crate::compiler::{
    Field,
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, MAX_SUPPORTED_SIGNED_BITS, MAX_SUPPORTED_UNSIGNED_BITS,
            OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessBitwiseOps {}

impl InstructionLoweringRule for LowerWitnessBitwiseOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_guarded_shift(b, context, *condition, inner.as_ref())
        } else {
            self.process_op(b, context, instruction)
        }
    }
}

impl LowerWitnessBitwiseOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_op(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        op: &OpCode,
    ) -> bool {
        let function_type_info = context.types();
        match op {
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor),
                result,
                lhs,
                rhs,
            } => {
                let lhs_witness = function_type_info.get_value_type(*lhs).is_witness_of();
                let rhs_witness = function_type_info.get_value_type(*rhs).is_witness_of();
                if lhs_witness || rhs_witness {
                    self.lower_binary_bitwise(
                        b,
                        function_type_info,
                        *kind,
                        *result,
                        *lhs,
                        *rhs,
                        lhs_witness,
                        rhs_witness,
                    );
                    true
                } else {
                    false
                }
            }
            OpCode::Not { result, value } => {
                self.lower_not(b, function_type_info, *result, *value);
                true
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } if integer_bits_and_signedness(context.types().get_value_type(*value)).is_some() => {
                self.lower_integer_sext(b, context, *result, *value, *from_bits, *to_bits);
                true
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
                result,
                lhs,
                rhs,
            } if context.types().get_value_type(*lhs).is_witness_of()
                || context.types().get_value_type(*rhs).is_witness_of() =>
            {
                self.lower_shift(b, context, None, *kind, *result, *lhs, *rhs);
                true
            }
            _ => false,
        }
    }

    fn process_guarded_shift(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        condition: ValueId,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
                result,
                lhs,
                rhs,
            } if context.types().get_value_type(*lhs).is_witness_of()
                || context.types().get_value_type(*rhs).is_witness_of() =>
            {
                self.lower_shift(b, context, Some(condition), *kind, *result, *lhs, *rhs);
                true
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_binary_bitwise(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let (bits, result_cast) =
            integer_bits_and_cast(function_type_info, result, "bitwise result");
        assert!(
            bits <= MAX_SUPPORTED_UNSIGNED_BITS,
            "bitwise spread width too large for natural-width Spread lowering: {bits}"
        );

        let lhs = b.cast_to(CastTarget::U(bits), lhs);
        let rhs = b.cast_to(CastTarget::U(bits), rhs);

        if bits == 1 {
            self.lower_u1_bitwise(b, kind, result, lhs, rhs);
            return;
        }

        let result_word = if bits == 64 {
            let lhs_limbs = decompose_u64_input(b, lhs, lhs_witness);
            let rhs_limbs = decompose_u64_input(b, rhs, rhs_witness);
            let result_limbs = lower_u64_limb_bitwise(b, kind, lhs_limbs, rhs_limbs);
            combine_u32_limbs(b, result_limbs)
        } else if bits == 128 {
            let lhs_limbs = extract_u128_limbs(b, lhs);
            let rhs_limbs = extract_u128_limbs(b, rhs);
            let lhs_lo = decompose_u64_input(b, lhs_limbs.lo, lhs_witness);
            let rhs_lo = decompose_u64_input(b, rhs_limbs.lo, rhs_witness);
            let lo = lower_u64_limb_bitwise(b, kind, lhs_lo, rhs_lo);
            let lhs_hi = decompose_u64_input(b, lhs_limbs.hi, lhs_witness);
            let rhs_hi = decompose_u64_input(b, rhs_limbs.hi, rhs_witness);
            let hi = lower_u64_limb_bitwise(b, kind, lhs_hi, rhs_hi);
            let lo = combine_u32_limbs(b, lo);
            let hi = combine_u32_limbs(b, hi);
            combine_u64_fields(b, lo, hi)
        } else {
            lower_word_bitwise(b, kind, lhs, rhs, bits as u8)
        };

        b.emit(OpCode::Cast {
            result,
            value: result_word,
            target: result_cast,
        });
    }

    fn lower_u1_bitwise(
        &self,
        b: &mut HLBlockEmitter<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let target = CastTarget::U(1);
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);

        let result_field = match kind {
            BinaryArithOpKind::And => b.mul(lhs_field, rhs_field),
            BinaryArithOpKind::Or => {
                let sum = b.add(lhs_field, rhs_field);
                let product = b.mul(lhs_field, rhs_field);
                b.sub(sum, product)
            }
            BinaryArithOpKind::Xor => {
                let sum = b.add(lhs_field, rhs_field);
                let two = b.field_const(Field::from(2));
                let product = b.mul(lhs_field, rhs_field);
                let two_product = b.mul(two, product);
                b.sub(sum, two_product)
            }
            _ => unreachable!(),
        };

        b.emit(OpCode::Cast {
            result,
            value: result_field,
            target,
        });
    }

    // FIELD-ASSUMPTION: L6-int-op-strategy
    // `not = (2^bits - 1) - value`. The all-ones mask `2^bits - 1` exceeds p at bits=64 on a
    // small field, so u64/u128 `not` must be done per-limb.
    fn lower_not(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
    ) {
        let (bits, cast_target) = integer_bits_and_cast(function_type_info, value, "bitwise not");
        // FIELD-ASSUMPTION: L4-decompose
        let ones = b.field_const((Field::from(2).pow([bits as u64])) - Field::ONE);
        let value_field = b.cast_to_field(value);
        let not_value = b.sub(ones, value_field);
        b.emit(OpCode::Cast {
            result,
            value: not_value,
            target: cast_target,
        });
    }

    // FIELD-ASSUMPTION: L6-int-op-strategy
    // Sign-extends via `value + sign * (two_pow(to_bits) - two_pow(from_bits))`. The
    // `two_pow(to_bits)` shift wraps mod p once `to_bits` reaches the field width.
    fn lower_integer_sext(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    ) {
        assert!(
            to_bits <= MAX_SUPPORTED_SIGNED_BITS,
            "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
        );
        let sign = if context.range(value).is_non_negative_in_signed(from_bits) {
            b.field_const(Field::ZERO)
        } else {
            let sign_bits = b.bit_range(value, from_bits - 1, 1);
            b.cast_to_field(sign_bits)
        };
        let value_field = b.cast_to_field(value);
        // FIELD-ASSUMPTION: L4-decompose
        let extension = b.field_const(two_pow(to_bits) - two_pow(from_bits));
        let offset = b.mul(sign, extension);
        let extended = b.add(value_field, offset);
        b.emit(OpCode::Cast {
            result,
            value: extended,
            target: cast_target_for_integer_type(context.types().get_value_type(result)),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_shift(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        assert!(!rhs_witness, "witness shift amounts are not supported");

        let bits = match lhs_type.strip_witness().expr {
            TypeExpr::U(bits) => bits,
            other => panic!("witness shift on unsupported lhs type {:?}", other),
        };

        let one_u = b.u_const(bits, 1);
        let factor = b.fresh_value();
        b.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Shl,
            result: factor,
            lhs: one_u,
            rhs,
        });

        match kind {
            BinaryArithOpKind::Shl => {
                let lhs_field = b.cast_to_field(lhs);
                let factor_field = b.cast_to_field(factor);
                let shifted = b.mul(lhs_field, factor_field);
                guarded_rangecheck(b, shifted, bits, guard);
                b.emit(OpCode::Cast {
                    result,
                    value: shifted,
                    target: CastTarget::U(bits),
                });
            }
            BinaryArithOpKind::Shr => {
                emit_guarded(
                    b,
                    guard,
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Div,
                        result,
                        lhs,
                        rhs: factor,
                    },
                );
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy)]
struct U64Limbs {
    lo: ValueId,
    hi: ValueId,
}

#[derive(Clone, Copy)]
struct U128Limbs {
    lo: ValueId,
    hi: ValueId,
}

// FIELD-ASSUMPTION: L4-decompose
// FIELD-ASSUMPTION: L4-two-pow
fn two_pow(exponent: usize) -> Field {
    Field::from(2).pow([exponent as u64])
}

fn guarded_rangecheck(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    bits: usize,
    guard: Option<ValueId>,
) {
    assert!(bits >= 1, "rangecheck width must be at least 1 bit");
    emit_guarded(
        b,
        guard,
        OpCode::Rangecheck {
            value,
            max_bits: bits,
        },
    );
}

fn emit_guarded(b: &mut HLBlockEmitter<'_>, guard: Option<ValueId>, op: OpCode) {
    if let Some(condition) = guard {
        b.emit(OpCode::Guard {
            condition,
            inner: Box::new(op),
        });
    } else {
        b.emit(op);
    }
}

fn cast_target_for_integer_type(ty: &Type) -> CastTarget {
    match ty.strip_witness().expr {
        TypeExpr::U(bits) => CastTarget::U(bits),
        TypeExpr::I(bits) => {
            assert!(
                bits <= MAX_SUPPORTED_SIGNED_BITS,
                "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
            );
            CastTarget::I(bits)
        }
        other => panic!("expected integer type, got {:?}", other),
    }
}

fn integer_bits_and_signedness(ty: &Type) -> Option<(usize, bool)> {
    match ty.strip_witness().expr {
        TypeExpr::U(bits) => Some((bits, false)),
        TypeExpr::I(bits) => Some((bits, true)),
        _ => None,
    }
}

fn integer_bits_and_cast(
    function_type_info: &FunctionTypeInfo,
    value: ValueId,
    context: &str,
) -> (usize, CastTarget) {
    match function_type_info
        .get_value_type(value)
        .strip_witness()
        .expr
    {
        TypeExpr::U(bits) => (bits, CastTarget::U(bits)),
        TypeExpr::I(bits) => {
            assert!(
                bits <= MAX_SUPPORTED_SIGNED_BITS,
                "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
            );
            (bits, CastTarget::I(bits))
        }
        other => panic!("{context}: expected integer type, got {:?}", other),
    }
}

fn spread_as_field(b: &mut impl HLEmitter, value: ValueId, bits: u8) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}

// FIELD-ASSUMPTION: L6-int-op-strategy
// Bitwise via spread-then-add: the spread of a `bits`-wide value occupies ~2*bits bits (cast
// to `U(bits*2)`), so on a ~64-bit field even a 32-bit spread nearly saturates p. Small fields
// need narrower spread limbs (why u64/u128 are already split into 32-bit limbs).
fn lower_word_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: ValueId,
    rhs: ValueId,
    bits: u8,
) -> ValueId {
    let lhs_spread = spread_as_field(b, lhs, bits);
    let rhs_spread = spread_as_field(b, rhs, bits);
    let input_spread_sum = b.add(lhs_spread, rhs_spread);
    let input_spread_sum = b.cast_to(CastTarget::U(bits as usize * 2), input_spread_sum);
    let (and_word, xor_word) = b.unspread(input_spread_sum, bits);

    match kind {
        BinaryArithOpKind::And => and_word,
        BinaryArithOpKind::Xor => xor_word,
        BinaryArithOpKind::Or => b.add(and_word, xor_word),
        _ => unreachable!(),
    }
}

fn lower_u64_limb_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: U64Limbs,
    rhs: U64Limbs,
) -> U64Limbs {
    U64Limbs {
        lo: lower_word_bitwise(b, kind, lhs.lo, rhs.lo, 32),
        hi: lower_word_bitwise(b, kind, lhs.hi, rhs.hi, 32),
    }
}

// FIELD-ASSUMPTION: L6-int-representation (combine_u32_limbs + combine_u64_fields)
// These recombine limbs into a single field cell (`lo + hi * 2^32` / `lo + hi * 2^64`). The
// 2^64 recombination exceeds p on a ~64-bit field, so wide results cannot live in one cell and
// must stay multi-cell; the shift width must derive from the field size.
fn combine_u32_limbs(b: &mut impl HLEmitter, limbs: U64Limbs) -> ValueId {
    let lo = b.cast_to_field(limbs.lo);
    let hi = b.cast_to_field(limbs.hi);
    let shift = b.field_const(Field::from(1u128 << 32));
    let shifted_hi = b.mul(hi, shift);
    b.add(lo, shifted_hi)
}

fn combine_u64_fields(b: &mut impl HLEmitter, lo: ValueId, hi: ValueId) -> ValueId {
    let lo = b.cast_to_field(lo);
    let hi = b.cast_to_field(hi);
    // FIELD-ASSUMPTION: L4-decompose
    let shift = b.field_const(two_pow(64));
    let shifted_hi = b.mul(hi, shift);
    b.add(lo, shifted_hi)
}

fn extract_u128_limbs(b: &mut impl HLEmitter, value: ValueId) -> U128Limbs {
    U128Limbs {
        lo: extract_u128_limb(b, value, 0),
        hi: extract_u128_limb(b, value, 64),
    }
}

fn extract_u128_limb(b: &mut impl HLEmitter, value: ValueId, offset: usize) -> ValueId {
    let limb = b.bit_range(value, offset, 64);
    b.cast_to(CastTarget::U(64), limb)
}

fn decompose_u64_input(b: &mut impl HLEmitter, value: ValueId, is_witness: bool) -> U64Limbs {
    if !is_witness {
        return extract_u64_limbs(b, value);
    }

    let pure_value = b.value_of(value);
    let hi_hint = extract_u64_limb(b, pure_value, 32);
    let hi_field = b.cast_to_field(hi_hint);
    let hi_wit = b.write_witness(hi_field);
    let lo = derive_low_u32_limb(b, value, hi_wit);

    U64Limbs {
        lo,
        hi: b.cast_to(CastTarget::U(32), hi_wit),
    }
}

fn extract_u64_limbs(b: &mut impl HLEmitter, value: ValueId) -> U64Limbs {
    U64Limbs {
        lo: extract_u64_limb(b, value, 0),
        hi: extract_u64_limb(b, value, 32),
    }
}

fn extract_u64_limb(b: &mut impl HLEmitter, value: ValueId, offset: usize) -> ValueId {
    let limb = b.bit_range(value, offset, 32);
    b.cast_to(CastTarget::U(32), limb)
}

fn derive_low_u32_limb(b: &mut impl HLEmitter, value: ValueId, hi_field: ValueId) -> ValueId {
    let value_field = b.cast_to_field(value);
    let shift = b.field_const(Field::from(1u128 << 32));
    let shifted_hi = b.mul(hi_field, shift);
    let lo_field = b.sub(value_field, shifted_hi);
    b.cast_to(CastTarget::U(32), lo_field)
}
