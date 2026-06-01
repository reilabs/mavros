use ark_ff::{AdditiveGroup as _, Field as _};

use crate::compiler::{
    Field,
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
                if guard.is_none() && !context.types().get_value_type(*value).is_witness_of() {
                    return false;
                }
                self.lower_assert_value(b, context, guard, *value);
                true
            }
            OpCode::AssertCmp { kind, lhs, rhs } => {
                let lhs_witness = context.types().get_value_type(*lhs).is_witness_of();
                let rhs_witness = context.types().get_value_type(*rhs).is_witness_of();
                if guard.is_none() && !lhs_witness && !rhs_witness {
                    return false;
                }
                match kind {
                    CmpKind::Eq => self.lower_assert_eq(b, context, guard, *lhs, *rhs),
                    CmpKind::Lt => self.lower_assert_lt(
                        b,
                        context,
                        guard,
                        *lhs,
                        *rhs,
                        lhs_witness,
                        rhs_witness,
                    ),
                }
                true
            }
            OpCode::AssertR1C { a, b: r1c_b, c } => {
                let a_witness = context.types().get_value_type(*a).is_witness_of();
                let b_witness = context.types().get_value_type(*r1c_b).is_witness_of();
                let c_witness = context.types().get_value_type(*c).is_witness_of();
                if guard.is_none() && !a_witness && !b_witness && !c_witness {
                    return false;
                }
                self.lower_assert_r1c(b, context, guard, *a, *r1c_b, *c);
                true
            }
            _ => false,
        }
    }

    fn lower_assert_value(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        value: ValueId,
    ) {
        let value_type = context.types().get_value_type(value);
        let value_field = b.ensure_field(value, &value_type.strip_witness());
        self.lower_assert_field(b, context, guard, value_field);
    }

    fn lower_assert_field(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        value_field: ValueId,
    ) {
        let cond_field = guard
            .map(|condition| b.ensure_field(condition, context.types().get_value_type(condition)))
            .unwrap_or_else(|| b.field_const(Field::ONE));
        b.constrain(cond_field, value_field, cond_field);
    }

    fn lower_assert_eq(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, &lhs_type.strip_witness());
        let rhs_field = b.ensure_field(rhs, &rhs_type.strip_witness());
        if let Some(condition) = guard {
            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            let diff = b.sub(lhs_field, rhs_field);
            let zero = b.field_const(Field::ZERO);
            b.constrain(cond_field, diff, zero);
        } else {
            let one = b.field_const(Field::ONE);
            b.constrain(lhs_field, one, rhs_field);
        }
    }

    fn lower_assert_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        match context.types().get_value_type(rhs).strip_witness().expr {
            TypeExpr::U(_) if guard.is_some() && !lhs_witness && !rhs_witness => {
                self.lower_assert_lt_via_cmp(b, context, guard, lhs, rhs);
            }
            TypeExpr::U(bits) => self.lower_unsigned_assert_lt(b, context, guard, lhs, rhs, bits),
            TypeExpr::I(_) => self.lower_assert_lt_via_cmp(b, context, guard, lhs, rhs),
            _ => panic!("ICE: AssertCmp Lt rhs is not an integer type"),
        }
    }

    fn lower_assert_lt_via_cmp(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let cmp = b.lt(lhs, rhs);
        let cmp_field = b.cast_to_field(cmp);
        self.lower_assert_field(b, context, guard, cmp_field);
    }

    fn lower_unsigned_assert_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        assert!(bits > 0, "rangecheck width must be at least 1 bit");
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, &lhs_type.strip_witness());
        let rhs_field = b.ensure_field(rhs, &rhs_type.strip_witness());
        let diff = b.sub(rhs_field, lhs_field);
        let one = b.field_const(Field::ONE);
        let diff_minus_one = b.sub(diff, one);
        self.emit_rangecheck(b, guard, diff_minus_one, bits);
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

    fn lower_assert_r1c(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        a: ValueId,
        r1c_b: ValueId,
        c: ValueId,
    ) {
        let a_type = context.types().get_value_type(a);
        let b_type = context.types().get_value_type(r1c_b);
        let c_type = context.types().get_value_type(c);
        let a_field = b.ensure_field(a, &a_type.strip_witness());
        let b_field = b.ensure_field(r1c_b, &b_type.strip_witness());
        let c_field = b.ensure_field(c, &c_type.strip_witness());

        if let Some(condition) = guard {
            let product = b.mul(a_field, b_field);
            let diff = b.sub(product, c_field);
            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            let zero = b.field_const(Field::ZERO);
            b.constrain(cond_field, diff, zero);
        } else {
            b.constrain(a_field, b_field, c_field);
        }
    }
}
