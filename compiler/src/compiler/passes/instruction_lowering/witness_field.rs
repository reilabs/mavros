use crate::compiler::ssa::{
    ValueId,
    hlssa::{
        BinaryArithOpKind, CastTarget, Endianness, LookupTarget, OpCode, Radix, SequenceTargetType,
        Type, TypeExpr,
        builder::{HLBlockEmitter, HLEmitter},
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
            self.process_op(b, context, Some(*condition), inner.as_ref())
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
            OpCode::ToBits {
                result,
                value,
                endianness,
                count,
            } => self.lower_to_bits(b, context, guard, *result, *value, *endianness, *count),
            OpCode::ToRadix {
                result,
                value,
                radix,
                endianness,
                count,
            } => self.lower_to_radix(
                b,
                context,
                guard,
                *result,
                *value,
                *radix,
                *endianness,
                *count,
            ),
            OpCode::Rangecheck { value, max_bits }
                if context.types().get_value_type(*value).is_witness_of() =>
            {
                self.lower_rangecheck(b, context, guard, *value, *max_bits);
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
        let one = b.field_const(b.field().one());
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
        let l_field = if l_type.strip_witness().is_field() {
            if_t
        } else {
            b.cast_to_field(if_t)
        };
        let r_field = if r_type.strip_witness().is_field() {
            if_f
        } else {
            b.cast_to_field(if_f)
        };

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

    fn lower_to_bits(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        endianness: Endianness,
        count: usize,
    ) -> bool {
        if !context.types().get_value_type(value).is_witness_of() {
            return false;
        }

        // Compute the decomposition in witgen, then bind each hinted bit to a fresh circuit
        // witness. WitnessWriteToFresh removes this hint chain from the R1CS pipeline later.
        let pure_value = b.value_of(value);
        // Witgen has one canonical representation for bit decomposition. Reverse the indices
        // below when the source operation requested big-endian output.
        let hint = b.to_bits(pure_value, Endianness::Little, count);
        let mut witnesses = vec![ValueId(0); count];

        let guard_field = guard
            .map(|condition| b.ensure_field(condition, context.types().get_value_type(condition)));
        let flag = guard_field.unwrap_or_else(|| b.field_const(Field::ONE));
        let rangecheck_type = LookupTarget::Rangecheck(1);
        let two = b.field_const(Field::from(2));
        let mut recomposed = b.field_const(Field::ZERO);

        // Horner evaluation must visit the most-significant output bit first. The output array
        // itself retains the endianness requested by the source program.
        let visit_order: Box<dyn Iterator<Item = usize>> = match endianness {
            Endianness::Little => Box::new((0..count).rev()),
            Endianness::Big => Box::new(0..count),
        };
        for i in visit_order {
            let hint_index = match endianness {
                Endianness::Little => i,
                Endianness::Big => count - i - 1,
            };
            let idx = b.u_const(32, hint_index as u128);
            let bit = b.array_get(hint, idx);
            let bit_field = b.cast_to_field(bit);
            let bit_witness = b.write_witness(bit_field);
            b.lookup_rngchk(rangecheck_type, bit_witness, flag);
            let shifted = b.mul(recomposed, two);
            recomposed = b.add(shifted, bit_witness);
            witnesses[i] = bit_witness;
        }

        // Bind the decomposition to the input. Under a guard, the equality and one-bit lookups
        // are active only when the guarded operation executes.
        if let Some(flag) = guard_field {
            let diff = b.sub(recomposed, value);
            let zero = b.field_const(Field::ZERO);
            b.constrain(diff, flag, zero);
        } else {
            let one = b.field_const(Field::ONE);
            b.constrain(recomposed, one, value);
        }

        let bit_elems = witnesses
            .into_iter()
            .map(|bit| b.cast_to(CastTarget::U(1), bit))
            .collect();
        b.emit(OpCode::MkSeq {
            result,
            elems: bit_elems,
            seq_type: SequenceTargetType::Array(count),
            elem_type: Type::witness_of(Type::u(1)),
        });
        true
    }

    fn lower_to_radix(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        radix: Radix<ValueId>,
        endianness: Endianness,
        count: usize,
    ) -> bool {
        let radix = match radix {
            Radix::Dyn(rv) => {
                let const_256 = b.u_const(32, 256);
                if let Some(condition) = guard {
                    b.emit(OpCode::Guard {
                        condition,
                        inner: Box::new(OpCode::AssertCmp {
                            kind: crate::compiler::ssa::hlssa::CmpKind::Eq,
                            lhs: rv,
                            rhs: const_256,
                        }),
                    });
                } else {
                    b.assert_eq(rv, const_256);
                }
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
        let mut current_sum = b.field_const(b.field().zero());
        let guard_field = guard
            .map(|condition| b.ensure_field(condition, context.types().get_value_type(condition)));
        let flag = guard_field.unwrap_or_else(|| b.field_const(b.field().one()));
        // `radix` is always `Bytes` here: a dynamic radix was asserted `== 256` and normalized to
        // `Bytes` above, so each digit is a static 8-bit rangecheck. No `DynRangecheck` is emitted.
        let radix_val = b.field_const(b.field().constant(256));
        let rangecheck_type = LookupTarget::Rangecheck(8);
        let visit_order: Box<dyn Iterator<Item = usize>> = match endianness {
            Endianness::Little => Box::new((0..count).rev()),
            Endianness::Big => Box::new(0..count),
        };
        for i in visit_order {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(hint, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            b.lookup_rngchk(rangecheck_type, byte_wit, flag);
            let shift_prev_res = b.mul(current_sum, radix_val);
            current_sum = b.add(shift_prev_res, byte_wit);
            witnesses[i] = byte_wit;
        }
        if let Some(flag) = guard_field {
            let diff = b.sub(current_sum, value);
            let zero = b.field_const(b.field().zero());
            b.constrain(diff, flag, zero);
        } else {
            let constrain_one = b.field_const(b.field().one());
            b.constrain(current_sum, constrain_one, value);
        }
        let byte_elems: Vec<ValueId> = witnesses
            .iter()
            .map(|&w| b.cast_to(CastTarget::U(8), w))
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
        let flag = guard
            .map(|condition| b.ensure_field(condition, context.types().get_value_type(condition)))
            .unwrap_or_else(|| b.field_const(b.field().one()));

        if max_bits == 0 {
            let zero = b.field_const(b.field().zero());
            b.constrain(flag, value_field, zero);
            return;
        }

        let max_bits: u8 = max_bits
            .try_into()
            .expect("rangecheck width must fit in LookupTarget::Rangecheck");
        b.lookup_rngchk(LookupTarget::Rangecheck(max_bits), value_field, flag);
    }
}

fn cast_target_for_integer_type(ty: &Type) -> CastTarget {
    match ty.strip_witness().expr {
        TypeExpr::U(bits) => CastTarget::U(bits),
        TypeExpr::I(bits) => CastTarget::I(bits),
        other => panic!("expected integer type, got {:?}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        pass_manager::{AnalysisStore, Pass},
        passes::instruction_lowering::InstructionLowering,
        ssa::{Terminator, hlssa::HLSSA},
    };

    fn lowered_to_bits(guarded: bool) -> (HLSSA, ValueId) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let value = ssa.fresh_value();
        let condition = guarded.then(|| ssa.fresh_value());
        let result = ssa.fresh_value();
        let function = ssa.get_unique_entrypoint_mut();
        function.add_return_type(Type::witness_of(Type::u(1)).array_of(3));
        let entry = function.get_entry_mut();
        entry.push_parameter(value, Type::witness_of(Type::field()));
        if let Some(condition) = condition {
            entry.push_parameter(condition, Type::witness_of(Type::u(1)));
        }
        let to_bits = OpCode::ToBits {
            result,
            value,
            endianness: Endianness::Little,
            count: 3,
        };
        entry.push_test_instruction(if let Some(condition) = condition {
            OpCode::Guard {
                condition,
                inner: Box::new(to_bits),
            }
        } else {
            to_bits
        });
        entry.set_terminator(Terminator::Return(vec![result]));

        InstructionLowering::witness_integer_ops().run(&mut ssa, &AnalysisStore::new());
        (ssa, result)
    }

    fn assert_constrained_bit_decomposition(ssa: &HLSSA, result: ValueId) {
        let instructions: Vec<_> = ssa
            .get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .collect();
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op, OpCode::ToBits { .. }))
                .count(),
            1,
            "the pure witgen hint must remain"
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op, OpCode::WriteWitness { .. }))
                .count(),
            3
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(
                    op,
                    OpCode::Lookup {
                        target: LookupTarget::Rangecheck(1),
                        ..
                    }
                ))
                .count(),
            3
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count(),
            1
        );
        assert!(instructions.iter().any(|op| matches!(
            op,
            OpCode::MkSeq {
                result: r,
                elems,
                seq_type: SequenceTargetType::Array(3),
                elem_type,
            } if *r == result && elems.len() == 3 && *elem_type == Type::witness_of(Type::u(1))
        )));
    }

    #[test]
    fn witnessed_to_bits_is_lowered_to_constrained_bit_witnesses() {
        let (ssa, result) = lowered_to_bits(false);
        assert_constrained_bit_decomposition(&ssa, result);
    }

    #[test]
    fn guarded_witnessed_to_bits_keeps_its_constraints() {
        let (ssa, result) = lowered_to_bits(true);
        assert_constrained_bit_decomposition(&ssa, result);
        assert!(
            ssa.get_unique_entrypoint()
                .get_entry()
                .get_instructions()
                .all(|op| !matches!(op, OpCode::Guard { .. }))
        );
    }
}
