//! Lowers witness-tainted bitwise and bit-selection operations before the main explicit-witness pass.
//!
//! This pass emits `Spread`/`Unspread` operations, except for `u64` bitwise ops where it keeps a
//! two-limb `u32` decomposition. It also canonicalizes witness integer casts/shifts into the shared
//! `BitRange` representation where possible.

use ark_ff::Field as _;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};

use crate::compiler::{
    Field,
    analysis::{flow_analysis::FlowAnalysis, types::FunctionTypeInfo},
    pass_manager::AnalysisId,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, OpCode, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{
    lowering_pass::{LoweringContext, LoweringPass},
    witness_integer_utils::{
        SignBitSource, cast_target_for_integer_type, emit_bit_range, extract_sign_bit,
        guarded_rangecheck, integer_bits_and_signedness, lower_unsigned_divmod,
        one_or_condition_field, two_pow,
    },
};

pub struct LowerWitnessBitwiseOps {}

impl LoweringPass for LowerWitnessBitwiseOps {
    const NAME: &'static str = "lower_witness_bitwise_ops";

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
            self.process_op(b, context, Some(condition), *inner);
        } else {
            self.process_op(b, context, None, instruction);
        }
    }
}

impl LowerWitnessBitwiseOps {
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

    fn process_op(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        op: OpCode,
    ) {
        let function_type_info = context.types();
        match op {
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor),
                result,
                lhs,
                rhs,
            } => {
                let lhs_witness = function_type_info.get_value_type(lhs).is_witness_of();
                let rhs_witness = function_type_info.get_value_type(rhs).is_witness_of();
                if lhs_witness || rhs_witness {
                    self.lower_binary_bitwise(
                        b,
                        function_type_info,
                        kind,
                        result,
                        lhs,
                        rhs,
                        lhs_witness,
                        rhs_witness,
                    );
                } else {
                    self.emit_guarded(
                        b,
                        guard,
                        OpCode::BinaryArithOp {
                            kind,
                            result,
                            lhs,
                            rhs,
                        },
                    );
                }
            }
            OpCode::Not { result, value } => {
                self.lower_not(b, function_type_info, result, value);
            }
            OpCode::Truncate {
                result,
                value,
                to_bits,
                from_bits,
            } if context.types().get_value_type(value).is_witness_of()
                && integer_bits_and_signedness(context.types().get_value_type(value)).is_some() =>
            {
                self.lower_integer_truncate(b, context, guard, result, value, to_bits, from_bits);
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } if context.types().get_value_type(value).is_witness_of()
                && integer_bits_and_signedness(context.types().get_value_type(value)).is_some() =>
            {
                self.lower_integer_sext(b, context, result, value, from_bits, to_bits);
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
                result,
                lhs,
                rhs,
            } if context.types().get_value_type(lhs).is_witness_of()
                || context.types().get_value_type(rhs).is_witness_of() =>
            {
                self.lower_shift(b, context, guard, kind, result, lhs, rhs);
            }
            other => self.emit_guarded(b, guard, other),
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
        let bits = unsigned_bits(function_type_info, lhs, "bitwise operand");
        assert!(
            bits <= 128,
            "bitwise spread width too large for natural-width Spread lowering: {bits}"
        );

        if bits == 1 {
            self.lower_u1_bitwise(b, kind, result, lhs, rhs);
            return;
        }

        let result_word = if bits == 64 {
            let lhs_limbs = decompose_u64_input(b, lhs, lhs_witness);
            let rhs_limbs = decompose_u64_input(b, rhs, rhs_witness);
            let result_limbs = lower_u64_limb_bitwise(b, kind, lhs_limbs, rhs_limbs);
            combine_u32_limbs(b, result_limbs)
        } else {
            lower_word_bitwise(b, kind, lhs, rhs, bits as u8)
        };

        b.emit(OpCode::Cast {
            result,
            value: result_word,
            target: CastTarget::U(bits),
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

    fn lower_not(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
    ) {
        let (bits, cast_target) = integer_bits_and_cast(function_type_info, value, "bitwise not");
        let ones = b.field_const((Field::from(2).pow([bits as u64])) - Field::ONE);
        let value_field = b.cast_to_field(value);
        let not_value = b.sub(ones, value_field);
        b.emit(OpCode::Cast {
            result,
            value: not_value,
            target: cast_target,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_integer_truncate(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        to_bits: usize,
        from_bits: usize,
    ) {
        if to_bits >= from_bits {
            self.emit_guarded(
                b,
                guard,
                OpCode::Truncate {
                    result,
                    value,
                    to_bits,
                    from_bits,
                },
            );
            return;
        }

        let low = emit_bit_range(b, value, 0, to_bits, None);
        b.emit(OpCode::Cast {
            result,
            value: low,
            target: cast_target_for_integer_type(context.types().get_value_type(result)),
        });
    }

    fn lower_integer_sext(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    ) {
        let value_field = b.cast_to_field(value);
        let sign = extract_sign_bit(
            b,
            value,
            from_bits,
            &context.range(value),
            SignBitSource::Integer,
        );
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
                let factor_range = context
                    .try_range(rhs)
                    .map(|range| {
                        let lo = range.lo().and_then(|v| v.to_u32()).unwrap_or(0);
                        let hi = range
                            .hi()
                            .and_then(|v| v.to_u32())
                            .unwrap_or(bits as u32 - 1)
                            .min(bits as u32 - 1);
                        crate::compiler::analysis::value_range_analysis::IntInterval::closed(
                            BigInt::one() << lo,
                            BigInt::one() << hi,
                        )
                    })
                    .unwrap_or_else(|| {
                        crate::compiler::analysis::value_range_analysis::IntInterval::closed(
                            BigInt::one(),
                            BigInt::one() << (bits as u32 - 1),
                        )
                    });
                let guard_is_witness = guard
                    .map(|condition| context.types().get_value_type(condition).is_witness_of())
                    .unwrap_or(false);
                let guard_flag = one_or_condition_field(b, context.types(), guard);
                let divmod = lower_unsigned_divmod(
                    b,
                    lhs,
                    factor,
                    bits,
                    true,
                    false,
                    &context.range(lhs),
                    &factor_range,
                    guard,
                    guard_is_witness,
                    guard_flag,
                );
                b.emit(OpCode::Cast {
                    result,
                    value: divmod.q,
                    target: CastTarget::U(bits),
                });
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

fn unsigned_bits(function_type_info: &FunctionTypeInfo, value: ValueId, context: &str) -> usize {
    match function_type_info
        .get_value_type(value)
        .strip_witness()
        .expr
    {
        TypeExpr::U(bits) => bits,
        other => panic!("{context}: expected unsigned integer type, got {:?}", other),
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
        TypeExpr::I(bits) => (bits, CastTarget::I(bits)),
        other => panic!("{context}: expected integer type, got {:?}", other),
    }
}

fn spread_as_field(b: &mut impl HLEmitter, value: ValueId, bits: u8) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}

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

fn combine_u32_limbs(b: &mut impl HLEmitter, limbs: U64Limbs) -> ValueId {
    let lo = b.cast_to_field(limbs.lo);
    let hi = b.cast_to_field(limbs.hi);
    let shift = b.field_const(Field::from(1u128 << 32));
    let shifted_hi = b.mul(hi, shift);
    b.add(lo, shifted_hi)
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
