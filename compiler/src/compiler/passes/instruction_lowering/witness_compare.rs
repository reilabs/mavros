use ark_ff::{AdditiveGroup as _, Field as _};

use crate::compiler::{
    Field,
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, CmpKind, OpCode, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessCompareOps {}

impl InstructionLoweringRule for LowerWitnessCompareOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_cmp(b, context, Some(*condition), inner.as_ref())
        } else {
            self.process_cmp(b, context, None, instruction)
        }
    }
}

impl LowerWitnessCompareOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_cmp(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        op: &OpCode,
    ) -> bool {
        let OpCode::Cmp {
            kind,
            result,
            lhs,
            rhs,
        } = op
        else {
            return false;
        };

        let lhs_witness = context.types().get_value_type(*lhs).is_witness_of();
        let rhs_witness = context.types().get_value_type(*rhs).is_witness_of();
        if !lhs_witness && !rhs_witness {
            return false;
        }

        match kind {
            CmpKind::Eq => self.lower_eq(
                b,
                context,
                guard,
                *result,
                *lhs,
                *rhs,
                lhs_witness,
                rhs_witness,
            ),
            CmpKind::Lt => self.lower_lt(
                b,
                context,
                guard,
                *result,
                *lhs,
                *rhs,
                lhs_witness,
                rhs_witness,
            ),
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_eq(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, &lhs_type.strip_witness());
        let rhs_field = b.ensure_field(rhs, &rhs_type.strip_witness());
        let diff = b.sub(lhs_field, rhs_field);

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let result_hint = b.eq(lhs_pure, rhs_pure);
        let result_hint = self.guard_hint(b, context, guard, result_hint);
        let result_hint_field = b.cast_to_field(result_hint);
        let result_witness = b.write_witness(result_hint_field);
        b.emit(OpCode::Cast {
            result,
            value: result_witness,
            target: CastTarget::U(1),
        });

        let diff_pure = b.value_of(diff);
        let one = b.field_const(Field::ONE);
        let div_hint = b.div(one, diff_pure);
        let div_hint_witness = b.write_witness(div_hint);

        let result_field = b.cast_to_field(result);
        let active = self.active_field(b, context, guard);
        let active_minus_result = b.sub(active, result_field);
        b.constrain(diff, div_hint_witness, active_minus_result);

        let zero = b.field_const(Field::ZERO);
        b.constrain(diff, result_field, zero);
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        match context.types().get_value_type(rhs).strip_witness().expr {
            TypeExpr::U(bits) => self.lower_unsigned_lt(
                b,
                context,
                guard,
                result,
                lhs,
                rhs,
                bits,
                lhs_witness,
                rhs_witness,
            ),
            TypeExpr::I(bits) => self.lower_signed_lt(
                b,
                context,
                guard,
                result,
                lhs,
                rhs,
                bits,
                lhs_witness,
                rhs_witness,
            ),
            _ => panic!("ICE: Cmp Lt rhs is not an integer type"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_signed_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let sign_l = self.sign_bit(b, lhs, bits);
        let sign_r = self.sign_bit(b, rhs, bits);
        let signs_differ = b.xor(sign_l, sign_r);

        let unsigned_result = b.fresh_value();
        self.lower_unsigned_lt(
            b,
            context,
            guard,
            unsigned_result,
            lhs,
            rhs,
            bits,
            lhs_witness,
            rhs_witness,
        );

        let signed_result = b.select(signs_differ, sign_l, unsigned_result);
        if let Some(condition) = guard {
            let zero = b.u_const(1, 0);
            b.emit(OpCode::Select {
                result,
                cond: condition,
                if_t: signed_result,
                if_f: zero,
            });
        } else {
            b.emit(OpCode::Cast {
                result,
                value: signed_result,
                target: CastTarget::U(1),
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_unsigned_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        assert!(bits > 0, "comparison width must be at least 1 bit");

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let lhs_hint = b.cast_to(CastTarget::U(bits), lhs_pure);
        let rhs_hint = b.cast_to(CastTarget::U(bits), rhs_pure);
        let result_hint = b.lt(lhs_hint, rhs_hint);
        let result_hint = self.guard_hint(b, context, guard, result_hint);
        let result_hint_field = b.cast_to_field(result_hint);
        let result_witness = b.write_witness(result_hint_field);
        b.emit(OpCode::Cast {
            result,
            value: result_witness,
            target: CastTarget::U(1),
        });

        let result_field = b.cast_to_field(result);
        self.emit_rangecheck(b, None, result_field, 1);
        self.constrain_result_inactive_zero(b, context, guard, result_field);

        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, &lhs_type.strip_witness());
        let rhs_field = b.ensure_field(rhs, &rhs_type.strip_witness());
        let one = b.field_const(Field::ONE);
        let true_delta = b.sub(rhs_field, lhs_field);
        let true_delta = b.sub(true_delta, one);
        let false_delta = b.sub(lhs_field, rhs_field);
        let delta_diff = b.sub(true_delta, false_delta);
        let selected_adjustment = b.mul(result_field, delta_diff);
        let selected_delta = b.add(false_delta, selected_adjustment);
        self.emit_rangecheck(b, guard, selected_delta, bits);
    }

    fn sign_bit(&self, b: &mut HLBlockEmitter<'_>, value: ValueId, bits: usize) -> ValueId {
        let sign = b.bit_range(value, bits - 1, 1);
        b.cast_to(CastTarget::U(1), sign)
    }

    fn guard_hint(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        value: ValueId,
    ) -> ValueId {
        if let Some(condition) = guard {
            let condition = if context.types().get_value_type(condition).is_witness_of() {
                b.value_of(condition)
            } else {
                condition
            };
            let zero = b.u_const(1, 0);
            b.select(condition, value, zero)
        } else {
            value
        }
    }

    fn active_field(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
    ) -> ValueId {
        if let Some(condition) = guard {
            b.ensure_field(condition, context.types().get_value_type(condition))
        } else {
            b.field_const(Field::ONE)
        }
    }

    fn constrain_result_inactive_zero(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result_field: ValueId,
    ) {
        if let Some(condition) = guard {
            let condition_field =
                b.ensure_field(condition, context.types().get_value_type(condition));
            let one = b.field_const(Field::ONE);
            let inactive = b.sub(one, condition_field);
            let zero = b.field_const(Field::ZERO);
            b.constrain(result_field, inactive, zero);
        }
    }

    fn emit_rangecheck(
        &self,
        b: &mut HLBlockEmitter<'_>,
        guard: Option<ValueId>,
        value: ValueId,
        bits: usize,
    ) {
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
}
