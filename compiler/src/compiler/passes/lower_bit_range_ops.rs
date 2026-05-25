//! Lowers canonical `BitRange` operations after the witness integer/bitwise passes have emitted
//! all bit selections.

use ark_ff::AdditiveGroup as _;

use crate::compiler::{
    Field,
    analysis::{flow_analysis::FlowAnalysis, types::FunctionTypeInfo},
    pass_manager::AnalysisId,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{
    lowering_pass::{LoweringContext, LoweringPass},
    witness_integer_utils::{guarded_rangecheck, one_or_condition_field, two_pow},
};

pub struct LowerBitRangeOps {}

impl LoweringPass for LowerBitRangeOps {
    const NAME: &'static str = "lower_bit_range_ops";

    fn preserved_analyses(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn process_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: OpCode,
    ) {
        match instruction {
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
                source_width,
            } => self.lower_bit_range(b, context, None, result, value, offset, width, source_width),
            OpCode::Guard { condition, inner } => match *inner {
                OpCode::BitRange {
                    result,
                    value,
                    offset,
                    width,
                    source_width,
                } => self.lower_bit_range(
                    b,
                    context,
                    Some(condition),
                    result,
                    value,
                    offset,
                    width,
                    source_width,
                ),
                other => b.emit(OpCode::Guard {
                    condition,
                    inner: Box::new(other),
                }),
            },
            other => b.emit(other),
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
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
        source_width: Option<usize>,
    ) {
        assert!(width > 0, "BitRange width must be at least 1");
        let value_type = context.types().get_value_type(value);
        let value_bits = value_type.get_bit_size();
        let source_bits = source_width.unwrap_or(value_bits);
        assert!(
            source_bits <= value_bits,
            "BitRange source width {} exceeds value width {}",
            source_bits,
            value_bits
        );
        assert!(
            offset + width <= source_bits,
            "BitRange({}, {}) exceeds source width {}",
            offset,
            width,
            source_bits
        );
        if value_type.strip_witness().is_field() {
            if value_type.is_witness_of() && source_width.is_some() {
                self.lower_witness_bit_range(
                    b,
                    context,
                    guard,
                    result,
                    value,
                    offset,
                    width,
                    source_bits,
                );
            } else {
                self.lower_witness_field_bit_range(b, guard, result, value, offset, width);
            }
            return;
        }

        if value_type.is_witness_of() {
            self.lower_witness_bit_range(
                b,
                context,
                guard,
                result,
                value,
                offset,
                width,
                source_bits,
            );
        } else {
            self.lower_pure_bit_range(b, context.types(), guard, result, value, offset, width);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_pure_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        types: &FunctionTypeInfo,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    ) {
        let value_type = types.get_value_type(value);
        let extracted = lower_pure_bit_range_value(b, value, value_type, offset, width);
        let target = cast_target_for_scalar_type(types.get_value_type(result));

        if let Some(condition) = guard {
            let casted = b.cast_to(target, extracted);
            let zero = zero_for_type(b, types.get_value_type(result));
            b.emit(OpCode::Select {
                result,
                cond: condition,
                if_t: casted,
                if_f: zero,
            });
        } else {
            b.emit(OpCode::Cast {
                result,
                value: extracted,
                target,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_witness_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
        source_bits: usize,
    ) {
        let value_type = context.types().get_value_type(value);
        let pure_value = b.value_of(value);
        let hint =
            lower_pure_bit_range_value(b, pure_value, &value_type.strip_witness(), offset, width);
        let hint_field = b.cast_to_field(hint);
        let hint_field = select_hint_when_guarded(b, context.types(), guard, hint_field);

        let result_witness = b.write_witness(hint_field);
        b.emit(OpCode::Cast {
            result,
            value: result_witness,
            target: cast_target_for_scalar_type(context.types().get_value_type(result)),
        });

        let result_field = b.cast_to_field(result);
        guarded_rangecheck(b, result_field, width, guard);

        let low = if offset == 0 {
            None
        } else {
            let low_hint =
                lower_pure_bit_range_value(b, pure_value, &value_type.strip_witness(), 0, offset);
            let low_hint = b.cast_to_field(low_hint);
            let low_hint = select_hint_when_guarded(b, context.types(), guard, low_hint);
            let low = b.write_witness(low_hint);
            guarded_rangecheck(b, low, offset, guard);
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
            let high_hint = select_hint_when_guarded(b, context.types(), guard, high_hint);
            let high = b.write_witness(high_hint);
            guarded_rangecheck(b, high, high_bits, guard);
            Some(high)
        };

        let mut reconstructed = low.unwrap_or_else(|| b.field_const(Field::ZERO));
        if offset > 0 {
            let result_shift = b.field_const(two_pow(offset));
            let result_shifted = b.mul(result_field, result_shift);
            reconstructed = b.add(reconstructed, result_shifted);
        } else {
            reconstructed = b.add(reconstructed, result_field);
        }
        if let Some(high) = high {
            let high_shift = b.field_const(two_pow(offset + width));
            let high_shifted = b.mul(high, high_shift);
            reconstructed = b.add(reconstructed, high_shifted);
        }

        let value_field = b.cast_to_field(value);
        let diff = b.sub(value_field, reconstructed);
        let zero = b.field_const(Field::ZERO);
        let flag = one_or_condition_field(b, context.types(), guard);
        b.constrain(flag, diff, zero);
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_witness_field_bit_range(
        &self,
        b: &mut HLBlockEmitter<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        offset: usize,
        width: usize,
    ) {
        let source_bits = 254;
        let end_bits = offset + width;
        let low_end = if end_bits == source_bits {
            value
        } else {
            let low_end = b.fresh_value();
            emit_guarded(
                b,
                guard,
                OpCode::Truncate {
                    result: low_end,
                    value,
                    to_bits: end_bits,
                    from_bits: source_bits,
                },
            );
            low_end
        };

        if offset == 0 {
            emit_guarded(
                b,
                guard,
                OpCode::Cast {
                    result,
                    value: low_end,
                    target: CastTarget::Field,
                },
            );
            return;
        }

        let low_start = b.fresh_value();
        emit_guarded(
            b,
            guard,
            OpCode::Truncate {
                result: low_start,
                value,
                to_bits: offset,
                from_bits: source_bits,
            },
        );
        let selected_shifted = b.sub(low_end, low_start);
        let divisor = b.field_const(two_pow(offset));
        emit_guarded(
            b,
            guard,
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Div,
                result,
                lhs: selected_shifted,
                rhs: divisor,
            },
        );
    }
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

fn lower_pure_bit_range_value(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    value_type: &Type,
    offset: usize,
    width: usize,
) -> ValueId {
    let source_bits = value_type.get_bit_size();
    match value_type.strip_witness().expr {
        TypeExpr::U(bits) | TypeExpr::I(bits) => {
            assert!(
                bits <= 128,
                "pure integer BitRange lowering only supports up to 128-bit integers"
            );
            let unsigned = b.cast_to(CastTarget::U(bits), value);
            let masked = if offset == 0 && width == source_bits {
                unsigned
            } else {
                let mask = bit_mask(bits, offset, width);
                let mask = b.u_const(bits, mask);
                let result = b.fresh_value();
                b.emit(OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::And,
                    result,
                    lhs: unsigned,
                    rhs: mask,
                });
                result
            };
            if offset == 0 {
                masked
            } else {
                let divisor = b.u_const(bits, 1u128 << offset);
                b.div(masked, divisor)
            }
        }
        TypeExpr::Field => {
            let low_width = offset + width;
            let low = if low_width == source_bits {
                value
            } else {
                b.truncate(value, low_width, source_bits)
            };
            if offset == 0 {
                low
            } else {
                let lower = b.truncate(value, offset, source_bits);
                let selected_shifted = b.sub(low, lower);
                let divisor = b.field_const(two_pow(offset));
                b.div(selected_shifted, divisor)
            }
        }
        other => panic!("BitRange expects a scalar source, got {:?}", other),
    }
}

fn bit_mask(bits: usize, offset: usize, width: usize) -> u128 {
    assert!(bits <= 128, "u128 mask cannot represent u{bits}");
    assert!(width > 0, "BitRange width must be at least 1");
    assert!(offset + width <= bits, "BitRange exceeds source width");
    if width == 128 {
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

fn zero_for_type(b: &mut HLBlockEmitter<'_>, ty: &Type) -> ValueId {
    match ty.strip_witness().expr {
        TypeExpr::Field => b.field_const(Field::ZERO),
        TypeExpr::U(bits) => b.u_const(bits, 0),
        TypeExpr::I(bits) => b.i_const(bits, 0),
        other => panic!("BitRange result must be scalar, got {:?}", other),
    }
}

fn select_hint_when_guarded(
    b: &mut HLBlockEmitter<'_>,
    types: &FunctionTypeInfo,
    guard: Option<ValueId>,
    hint_field: ValueId,
) -> ValueId {
    if let Some(condition) = guard {
        let condition = if types.get_value_type(condition).is_witness_of() {
            b.value_of(condition)
        } else {
            condition
        };
        let zero = b.field_const(Field::ZERO);
        b.select(condition, hint_field, zero)
    } else {
        hint_field
    }
}
