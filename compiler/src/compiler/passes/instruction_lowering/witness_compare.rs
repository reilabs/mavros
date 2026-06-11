use ark_ff::Field as _;

use crate::compiler::{
    Field,
    ssa::{
        ValueId,
        hlssa::{
            CmpKind, OpCode, Type, TypeExpr,
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
        self.process_cmp(b, context, instruction)
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
            CmpKind::Eq => self.lower_eq(b, context, *result, *lhs, *rhs, lhs_witness, rhs_witness),
            CmpKind::Lt => self.lower_lt(b, context, *result, *lhs, *rhs, lhs_witness, rhs_witness),
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_eq(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, lhs_type);
        let rhs_field = b.ensure_field(rhs, rhs_type);
        let diff = b.sub(lhs_field, rhs_field);

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let result_hint = b.eq(lhs_pure, rhs_pure);
        let result_hint_field = b.cast_to_field(result_hint);
        let result_witness = b.write_witness(result_hint_field);
        b.emit(OpCode::Cast {
            result,
            value: result_witness,
            target: Type::witness_of(Type::u(1)),
        });

        let result_field = b.ensure_field(result, &Type::witness_of(Type::u(1)));
        let one = b.field_const(Field::ONE);
        let not_result = b.sub(one, result_field);
        let quotient = b.div(not_result, diff);
        let quotient_plus_result = b.add(quotient, result_field);
        b.constrain(diff, quotient_plus_result, not_result);
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        match context.types().get_value_type(rhs).strip_witness().expr {
            TypeExpr::U(bits) => {
                self.lower_unsigned_lt(b, context, result, lhs, rhs, bits, lhs_witness, rhs_witness)
            }
            TypeExpr::I(bits) => {
                self.lower_signed_lt(b, context, result, lhs, rhs, bits, lhs_witness, rhs_witness)
            }
            _ => panic!("ICE: Cmp Lt rhs is not an integer type"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_signed_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let sign_l = self.sign_bit(b, lhs, bits, lhs_witness);
        let sign_r = self.sign_bit(b, rhs, bits, rhs_witness);
        let signs_differ = b.xor(sign_l, sign_r);

        let unsigned_result = b.fresh_value();
        self.lower_unsigned_lt(
            b,
            context,
            unsigned_result,
            lhs,
            rhs,
            bits,
            lhs_witness,
            rhs_witness,
        );

        let signed_result = b.select(signs_differ, sign_l, unsigned_result);
        b.emit(OpCode::Cast {
            result,
            value: signed_result,
            target: Type::witness_of(Type::u(1)),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_unsigned_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
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
        let lhs_hint = b.cast_to(Type::u(bits), lhs_pure);
        let rhs_hint = b.cast_to(Type::u(bits), rhs_pure);
        let result_hint = b.lt(lhs_hint, rhs_hint);
        let result_hint_field = b.cast_to_field(result_hint);
        let result_witness = b.write_witness(result_hint_field);
        b.emit(OpCode::Cast {
            result,
            value: result_witness,
            target: Type::witness_of(Type::u(1)),
        });

        let result_field = b.ensure_field(result, &Type::witness_of(Type::u(1)));
        self.emit_rangecheck(b, result_field, 1);

        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, lhs_type);
        let rhs_field = b.ensure_field(rhs, rhs_type);
        let one = b.field_const(Field::ONE);
        let true_delta = b.sub(rhs_field, lhs_field);
        let true_delta = b.sub(true_delta, one);
        let false_delta = b.sub(lhs_field, rhs_field);
        let delta_diff = b.sub(true_delta, false_delta);
        let selected_adjustment = b.mul(result_field, delta_diff);
        let selected_delta = b.add(false_delta, selected_adjustment);
        self.emit_rangecheck(b, selected_delta, bits);
    }

    fn sign_bit(
        &self,
        b: &mut HLBlockEmitter<'_>,
        value: ValueId,
        bits: usize,
        is_witness: bool,
    ) -> ValueId {
        let sign = b.bit_range(value, bits - 1, 1);
        let target = if is_witness {
            Type::witness_of(Type::u(1))
        } else {
            Type::u(1)
        };
        b.cast_to(target, sign)
    }

    fn emit_rangecheck(&self, b: &mut HLBlockEmitter<'_>, value: ValueId, bits: usize) {
        b.emit(OpCode::Rangecheck {
            value,
            max_bits: bits,
        });
    }
}
