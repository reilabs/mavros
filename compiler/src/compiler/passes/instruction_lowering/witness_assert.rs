use ark_ff::{AdditiveGroup as _, Field as _};

use crate::compiler::{
    Field,
    passes::witness_algebra,
    ssa::{
        ValueId,
        hlssa::{
            CmpKind, OpCode, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessAssertOps {}

impl InstructionLoweringRule for LowerWitnessAssertOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_assert(b, context, Some(*condition), inner.as_ref())
        } else {
            self.process_assert(b, context, None, instruction)
        }
    }
}

impl LowerWitnessAssertOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_assert(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::Assert { value } => {
                let is_witness = context.types().get_value_type(*value).is_witness_of();
                if guard.is_none() && !is_witness {
                    return false;
                }
                self.lower_assert(b, context, guard, *value);
                true
            }
            OpCode::AssertCmp { kind, lhs, rhs } => {
                let l_type = context.types().get_value_type(*lhs);
                let r_type = context.types().get_value_type(*rhs);
                let l_taint = l_type.is_witness_of();
                let r_taint = r_type.is_witness_of();
                if guard.is_none() && !l_taint && !r_taint {
                    return false;
                }
                self.lower_assert_cmp(b, context, guard, *kind, *lhs, *rhs, l_taint, r_taint);
                true
            }
            OpCode::AssertR1C { a, b: r1c_b, c } => {
                let a_taint = context.types().get_value_type(*a).is_witness_of();
                let b_taint = context.types().get_value_type(*r1c_b).is_witness_of();
                let c_taint = context.types().get_value_type(*c).is_witness_of();
                if guard.is_none() && !a_taint && !b_taint && !c_taint {
                    return false;
                }
                self.lower_assert_r1c(b, context, guard, *a, *r1c_b, *c, a_taint, b_taint, c_taint);
                true
            }
            _ => false,
        }
    }

    fn lower_assert(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        value: ValueId,
    ) {
        let v_type = context.types().get_value_type(value);
        let one = b.field_const(Field::ONE);
        let v_field = b.ensure_field(value, &v_type.strip_witness());
        if let Some(condition) = guard {
            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            b.constrain(cond_field, v_field, cond_field);
        } else {
            b.constrain(v_field, one, one);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_assert_cmp(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: CmpKind,
        lhs: ValueId,
        rhs: ValueId,
        l_taint: bool,
        r_taint: bool,
    ) {
        match kind {
            CmpKind::Eq => self.lower_assert_eq(b, context, guard, lhs, rhs),
            CmpKind::Lt => self.lower_assert_lt(b, context, guard, lhs, rhs, l_taint, r_taint),
        }
    }

    fn lower_assert_eq(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let l_type = context.types().get_value_type(lhs);
        let r_type = context.types().get_value_type(rhs);
        let l = b.ensure_field(lhs, &l_type.strip_witness());
        let r = b.ensure_field(rhs, &r_type.strip_witness());
        if let Some(condition) = guard {
            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            let diff = b.sub(l, r);
            let zero = b.field_const(Field::ZERO);
            b.constrain(cond_field, diff, zero);
        } else {
            let one = b.field_const(Field::ONE);
            b.constrain(l, one, r);
        }
    }

    fn lower_assert_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
        l_taint: bool,
        r_taint: bool,
    ) {
        if guard.is_some() && !l_taint && !r_taint {
            self.lower_guarded_pure_lt(b, context, guard.unwrap(), lhs, rhs);
            return;
        }

        let rhs_stripped = context.types().get_value_type(rhs).strip_witness().expr;
        let (bits, is_signed) = match rhs_stripped {
            TypeExpr::U(bits) => (bits, false),
            TypeExpr::I(bits) => (bits, true),
            _ => panic!("ICE: AssertCmp Lt rhs is not an integer type"),
        };

        if is_signed || guard.is_some() {
            let result = b.fresh_value();
            let l_range = context.range(lhs);
            let r_range = context.range(rhs);
            witness_algebra::lower_witness_lt(
                b,
                context.types(),
                lhs,
                rhs,
                result,
                l_taint,
                r_taint,
                &l_range,
                &r_range,
            );
            self.constrain_assert_result(b, context, guard, result);
        } else {
            let l_pure = if l_taint { b.value_of(lhs) } else { lhs };
            let r_pure = if r_taint { b.value_of(rhs) } else { rhs };
            let l_field_pure = b.cast_to_field(l_pure);
            let r_field_pure = b.cast_to_field(r_pure);
            let diff_hint = b.sub(r_field_pure, l_field_pure);
            let one = b.field_const(Field::ONE);
            let diff_minus_one_hint = b.sub(diff_hint, one);
            let diff_wit = b.write_witness(diff_minus_one_hint);
            let flag = b.field_const(Field::ONE);
            witness_algebra::gen_witness_rangecheck_bits(b, diff_wit, bits, flag);
        }
    }

    fn lower_guarded_pure_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        condition: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let cmp = b.lt(lhs, rhs);
        let cmp_field = b.cast_to_field(cmp);
        let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
        b.constrain(cond_field, cmp_field, cond_field);
    }

    fn constrain_assert_result(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
    ) {
        let result_field = b.cast_to_field(result);
        let one = b.field_const(Field::ONE);
        if let Some(condition) = guard {
            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            b.constrain(cond_field, result_field, cond_field);
        } else {
            b.constrain(result_field, one, one);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_assert_r1c(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        a: ValueId,
        r1c_b: ValueId,
        c: ValueId,
        a_taint: bool,
        b_taint: bool,
        _c_taint: bool,
    ) {
        if let Some(condition) = guard {
            let a_pure = if a_taint { b.value_of(a) } else { a };
            let b_pure = if b_taint { b.value_of(r1c_b) } else { r1c_b };
            let a_pure = b.ensure_field(a_pure, &context.types().get_value_type(a).strip_witness());
            let b_pure = b.ensure_field(
                b_pure,
                &context.types().get_value_type(r1c_b).strip_witness(),
            );
            let product_hint = b.mul(a_pure, b_pure);
            let product_wit = b.write_witness(product_hint);
            let a_field = b.ensure_field(a, &context.types().get_value_type(a).strip_witness());
            let b_field = b.ensure_field(
                r1c_b,
                &context.types().get_value_type(r1c_b).strip_witness(),
            );
            b.constrain(a_field, b_field, product_wit);

            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            let c_field = b.ensure_field(c, &context.types().get_value_type(c).strip_witness());
            let diff = b.sub(product_wit, c_field);
            let zero = b.field_const(Field::ZERO);
            b.constrain(cond_field, diff, zero);
        } else {
            b.constrain(a, r1c_b, c);
        }
    }
}
