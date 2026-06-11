use ark_ff::{AdditiveGroup as _, Field as _};

use crate::compiler::{
    Field,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, Endianness, LookupTarget, OpCode, Radix, SequenceTargetType, Type,
            TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessFieldOps {}

impl InstructionLoweringRule for LowerWitnessFieldOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_guarded_op(b, context, *condition, inner.as_ref())
        } else {
            self.process_op(b, context, None, instruction)
        }
    }
}

impl LowerWitnessFieldOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_op(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result,
                lhs,
                rhs,
            } => self.lower_divmod(b, context, guard, *kind, *result, *lhs, *rhs),
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } if context.types().get_value_type(*cond).is_witness_of() => {
                self.lower_select(b, context, *result, *cond, *if_t, *if_f);
                true
            }
            OpCode::ToRadix {
                result,
                value,
                radix,
                endianness,
                count,
            } => self.lower_to_radix(b, context, *result, *value, *radix, *endianness, *count),
            OpCode::Rangecheck { value, max_bits }
                if context.types().get_value_type(*value).is_witness_of() =>
            {
                self.lower_rangecheck(b, context, guard, *value, *max_bits);
                true
            }
            _ => false,
        }
    }

    fn process_guarded_op(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        condition: ValueId,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result,
                lhs,
                rhs,
            } => self.lower_divmod(b, context, Some(condition), *kind, *result, *lhs, *rhs),
            OpCode::Rangecheck { value, max_bits }
                if context.types().get_value_type(*value).is_witness_of() =>
            {
                self.lower_rangecheck(b, context, Some(condition), *value, *max_bits);
                true
            }
            _ => false,
        }
    }

    fn lower_divmod(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let lhs_type = context.types().get_value_type(lhs);
        if !lhs_type.strip_witness().is_field() {
            return false;
        }

        let lhs_witness = lhs_type.is_witness_of();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();
        if !lhs_witness && !rhs_witness {
            return false;
        }

        assert!(
            kind == BinaryArithOpKind::Div,
            "Modulo is not defined on field elements"
        );

        if let Some(condition) = guard {
            self.lower_field_div_guarded(
                b,
                context,
                condition,
                result,
                lhs,
                rhs,
                lhs_witness,
                rhs_witness,
            );
            return true;
        }

        self.lower_field_div(b, result, lhs, rhs, lhs_witness, rhs_witness)
    }

    fn lower_field_div(
        &self,
        b: &mut HLBlockEmitter<'_>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) -> bool {
        if lhs_witness && !rhs_witness {
            return false;
        }

        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };
        let quotient_hint = b.div(lhs_pure, rhs_pure);
        let quotient_hint_field = b.cast_to_field(quotient_hint);
        b.emit(OpCode::WriteWitness {
            result: Some(result),
            value: quotient_hint_field,
            pinned: false,
        });
        b.constrain(result, rhs, lhs);
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_field_div_guarded(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        condition: ValueId,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        if lhs_witness && !rhs_witness {
            b.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Div,
                result,
                lhs,
                rhs,
            });
            return;
        }

        let condition_type = context.types().get_value_type(condition);
        let condition_field = b.ensure_field(condition, condition_type);
        let condition_pure = if condition_type.is_witness_of() {
            b.value_of(condition_field)
        } else {
            condition_field
        };
        let lhs_pure = if lhs_witness { b.value_of(lhs) } else { lhs };
        let rhs_pure = if rhs_witness { b.value_of(rhs) } else { rhs };

        let lhs_gated_hint = b.mul(lhs_pure, condition_pure);
        let one = b.field_const(Field::ONE);
        let one_minus_condition = b.sub(one, condition_pure);
        let rhs_when_active = b.mul(rhs_pure, condition_pure);
        let safe_rhs_hint = b.add(rhs_when_active, one_minus_condition);
        let quotient_hint = b.div(lhs_gated_hint, safe_rhs_hint);
        let quotient_hint_field = b.cast_to_field(quotient_hint);
        b.emit(OpCode::WriteWitness {
            result: Some(result),
            value: quotient_hint_field,
            pinned: false,
        });

        let lhs_gated = if lhs_witness && condition_type.is_witness_of() {
            let lhs_gated_witness = b.write_witness(lhs_gated_hint);
            b.constrain(lhs, condition_field, lhs_gated_witness);
            lhs_gated_witness
        } else {
            b.mul(lhs, condition_field)
        };
        b.constrain(result, rhs, lhs_gated);
    }

    fn lower_select(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        cond: ValueId,
        if_t: ValueId,
        if_f: ValueId,
    ) {
        let l_type = context.types().get_value_type(if_t);
        let r_type = context.types().get_value_type(if_f);
        let l_field = b.ensure_field(if_t, l_type);
        let r_field = b.ensure_field(if_f, r_type);

        let l_sub_r = b.sub(l_field, r_field);
        let cond_field = b.ensure_field(cond, context.types().get_value_type(cond));
        let cond_times_diff = b.mul(l_sub_r, cond_field);
        let result_type = context.types().get_value_type(result);
        if result_type.strip_witness().is_field() {
            b.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result,
                lhs: cond_times_diff,
                rhs: r_field,
            });
        } else {
            let selected = b.add(cond_times_diff, r_field);
            b.emit(OpCode::Cast {
                result,
                value: selected,
                target: cast_target_for_integer_type(result_type),
            });
        }
    }

    fn lower_to_radix(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        radix: Radix<ValueId>,
        endianness: Endianness,
        count: usize,
    ) -> bool {
        let radix = match radix {
            Radix::Dyn(rv) => {
                let const_256 = b.u_const(32, 256);
                b.assert_eq(rv, const_256);
                Radix::Bytes
            }
            Radix::Bytes if !context.types().get_value_type(value).is_witness_of() => {
                return false;
            }
            Radix::Bytes => Radix::Bytes,
        };

        if !context.types().get_value_type(value).is_witness_of() {
            b.emit(OpCode::ToRadix {
                result,
                value,
                radix,
                endianness,
                count,
            });
            return true;
        }

        let pure_value = b.value_of(value);
        let hint = b.to_radix(pure_value, radix, endianness, count);
        let mut witnesses = vec![ValueId(0); count];
        let mut current_sum = b.field_const(Field::ZERO);
        let radix_val = match radix {
            Radix::Bytes => b.field_const(Field::from(256)),
            Radix::Dyn(radix) => b.cast_to(Type::field(), radix),
        };
        let rangecheck_type = match radix {
            Radix::Bytes => LookupTarget::Rangecheck(8),
            Radix::Dyn(radix) => LookupTarget::DynRangecheck(radix),
        };
        let visit_order: Box<dyn Iterator<Item = usize>> = match endianness {
            Endianness::Little => Box::new((0..count).rev()),
            Endianness::Big => Box::new(0..count),
        };
        for i in visit_order {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(hint, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            let one = b.field_const(Field::ONE);
            b.lookup_rngchk(rangecheck_type, byte_wit, one);
            let shift_prev_res = b.mul(current_sum, radix_val);
            current_sum = b.add(shift_prev_res, byte_wit);
            witnesses[i] = byte_wit;
        }
        let constrain_one = b.field_const(Field::ONE);
        b.constrain(current_sum, constrain_one, value);
        let byte_elems: Vec<ValueId> = witnesses
            .iter()
            .map(|&w| b.cast_to(Type::witness_of(Type::u(8)), w))
            .collect();
        b.emit(OpCode::MkSeq {
            result,
            elems: byte_elems,
            seq_type: SequenceTargetType::Array(count),
            elem_type: Type::witness_of(Type::u(8)),
        });
        true
    }

    fn lower_rangecheck(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        value: ValueId,
        max_bits: usize,
    ) {
        let value_field = b.ensure_field(value, context.types().get_value_type(value));
        if max_bits == 0 {
            let flag = guard
                .map(|condition| {
                    b.ensure_field(condition, context.types().get_value_type(condition))
                })
                .unwrap_or_else(|| b.field_const(Field::ONE));
            let zero = b.field_const(Field::ZERO);
            b.constrain(flag, value_field, zero);
            return;
        }

        let max_bits: u8 = max_bits
            .try_into()
            .expect("rangecheck width must fit in LookupTarget::Rangecheck");
        let flag = guard
            .map(|condition| b.ensure_field(condition, context.types().get_value_type(condition)))
            .unwrap_or_else(|| b.field_const(Field::ONE));
        b.lookup_rngchk(LookupTarget::Rangecheck(max_bits), value_field, flag);
    }
}

fn cast_target_for_integer_type(ty: &Type) -> Type {
    match ty.strip_witness().expr {
        TypeExpr::U(_) | TypeExpr::I(_) => ty.clone(),
        other => panic!("expected integer type, got {:?}", other),
    }
}
