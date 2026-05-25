use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};

use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
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
        assign_field, cast_target_for_integer_type, extract_sign_bit_from_integer,
        guarded_or_zero_field, guarded_rangecheck, integer_bits_and_signedness,
        lower_unsigned_divmod, one_or_condition_field, two_pow, two_pow_u128,
    },
};

pub struct LowerWitnessIntegerCastsAndShifts {}

impl LoweringPass for LowerWitnessIntegerCastsAndShifts {
    const NAME: &'static str = "lower_witness_integer_casts_and_shifts";

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

impl LowerWitnessIntegerCastsAndShifts {
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
        match op {
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
                self.lower_integer_sext(b, context, guard, result, value, from_bits, to_bits);
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
    fn lower_integer_truncate(
        &self,
        b: &mut HLBlockEmitter<'_>,
        _context: &LoweringContext<'_>,
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

        let value_field = b.cast_to_field(value);
        let value_for_hint = guarded_or_zero_field(b, value_field, guard);
        let pure_value = b.value_of(value_for_hint);
        let unsigned_value = b.cast_to(CastTarget::U(from_bits), pure_value);
        let divisor = b.u_const(from_bits, two_pow_u128(to_bits));
        let hi_hint = b.div(unsigned_value, divisor);
        let hi_hint_field = b.cast_to_field(hi_hint);
        let hi_wit = b.write_witness(hi_hint_field);

        let shift = b.field_const(two_pow(to_bits));
        let shifted_hi = b.mul(hi_wit, shift);
        let lo = b.sub(value_field, shifted_hi);

        if to_bits > 0 {
            guarded_rangecheck(b, lo, to_bits, guard);
        }
        let hi_bits = from_bits - to_bits;
        if hi_bits > 0 {
            guarded_rangecheck(b, hi_wit, hi_bits, guard);
        }

        let lo = guarded_or_zero_field(b, lo, guard);
        assign_field(b, result, lo);
    }

    fn lower_integer_sext(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    ) {
        let value_type = context.types().get_value_type(value);
        let value_field = b.cast_to_field(value);
        let sign = extract_sign_bit_from_integer(
            b,
            value,
            value_field,
            from_bits,
            true,
            &context.range(value),
            guard,
        );
        let extension = b.field_const(two_pow(to_bits) - two_pow(from_bits));
        let offset = b.mul(sign, extension);
        let extended = b.add(value_field, offset);
        b.emit(OpCode::Cast {
            result,
            value: extended,
            target: cast_target_for_integer_type(value_type).with_bits(to_bits),
        });
    }

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

trait CastTargetExt {
    fn with_bits(self, bits: usize) -> CastTarget;
}

impl CastTargetExt for CastTarget {
    fn with_bits(self, bits: usize) -> CastTarget {
        match self {
            CastTarget::U(_) => CastTarget::U(bits),
            CastTarget::I(_) => CastTarget::I(bits),
            other => other,
        }
    }
}
