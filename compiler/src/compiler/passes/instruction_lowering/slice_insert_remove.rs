//! Lowers the remaining (pure-length) `SliceInsert`/`SliceRemove` into a bounds assert plus a
//! rebuild scan

use crate::compiler::ssa::{
    ValueId,
    hlssa::{
        CastTarget, CmpKind, OpCode, SequenceTargetType, SliceOpDir, Type,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

use super::{
    InstructionLoweringRule, LoweringContext,
    witness_array::{
        merge_select_for_slice_leaves, push_witness_of_to_leaves_for_slice_children, uint_bits,
    },
};

fn unwrap_guard(instruction: &OpCode) -> (Option<ValueId>, &OpCode) {
    match instruction {
        OpCode::Guard { condition, inner } => (Some(*condition), inner.as_ref()),
        other => (None, other),
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

pub struct LowerSliceInsert {}

impl LowerSliceInsert {
    pub fn new() -> Self {
        Self {}
    }
}

impl InstructionLoweringRule for LowerSliceInsert {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (guard, op) = unwrap_guard(instruction);
        let OpCode::SliceInsert {
            result,
            slice,
            index,
            value,
        } = op
        else {
            return false;
        };

        let elem_ty = context.types().get_value_type(*slice).get_array_element();
        let index_ty = context.types().get_value_type(*index).clone();
        let value_ty = context.types().get_value_type(*value).clone();
        let mut sel_elem_ty = Type::join(&elem_ty, &value_ty);
        if index_ty.is_witness_of() {
            sel_elem_ty = push_witness_of_to_leaves_for_slice_children(&sel_elem_ty);
        }
        let acc_slice_ty = sel_elem_ty.clone().slice_of();

        let idx_bits = uint_bits(&index_ty, "slice insert index");
        let len = b.slice_len(*slice);
        let zero = b.u_const(32, 0);
        let one = b.u_const(32, 1);
        let new_len = b.add(len, one);

        emit_guarded(
            b,
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::Lt,
                lhs: *index,
                rhs: new_len,
            },
        );

        let empty = b.mk_seq(vec![], SequenceTargetType::Slice, sel_elem_ty.clone());
        let idx_v = *index;
        let value_v = *value;
        let slice_v = *slice;

        let grown = b.slice_push(slice_v, vec![value_v], SliceOpDir::Back);
        let index_is_witness = index_ty.is_witness_of();
        let rebuilt = b.build_slice_extend_loop(new_len, (empty, acc_slice_ty), move |b, i| {
            // `i - 1` computed as `i - (i > 0)`
            let i_is_positive = b.lt(zero, i);
            let dec = b.cast_to(CastTarget::U(32), i_is_positive);
            let prev = b.sub(i, dec);
            let cmp_i = if idx_bits == 32 {
                i
            } else {
                b.cast_to(CastTarget::U(idx_bits), i)
            };
            let below = b.lt(cmp_i, idx_v);
            let at = b.eq(cmp_i, idx_v);
            if index_is_witness {
                let cur_val = b.array_get(grown, i);
                let prev_val = b.array_get(grown, prev);
                let _temp = merge_select_for_slice_leaves(b, at, value_v, prev_val, &sel_elem_ty);
                merge_select_for_slice_leaves(b, below, cur_val, _temp, &sel_elem_ty)
            } else {
                // Multiplexer
                let below32 = b.cast_to(CastTarget::U(32), below);
                let at32 = b.cast_to(CastTarget::U(32), at);
                let _tempsum = b.add(below32, at32);
                let above32 = b.sub(one, _tempsum);
                let t1 = b.mul(below32, i);
                let t2 = b.mul(at32, len);
                let t3 = b.mul(above32, prev);
                let _temp = b.add(t1, t2);
                let src = b.add(_temp, t3);
                b.array_get(grown, src)
            }
        });
        b.emit(OpCode::Cast {
            result: *result,
            value: rebuilt,
            target: CastTarget::Nop,
        });
        true
    }
}

pub struct LowerSliceRemove {}

impl LowerSliceRemove {
    pub fn new() -> Self {
        Self {}
    }
}

impl InstructionLoweringRule for LowerSliceRemove {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (guard, op) = unwrap_guard(instruction);
        let OpCode::SliceRemove {
            result_slice,
            result_elem,
            slice,
            index,
        } = op
        else {
            return false;
        };

        let elem_ty = context.types().get_value_type(*slice).get_array_element();
        let index_ty = context.types().get_value_type(*index).clone();
        let mut sel_elem_ty = elem_ty.clone();
        if index_ty.is_witness_of() {
            sel_elem_ty = push_witness_of_to_leaves_for_slice_children(&sel_elem_ty);
        }
        let acc_slice_ty = sel_elem_ty.clone().slice_of();

        let idx_bits = uint_bits(&index_ty, "slice remove index");
        let len = b.slice_len(*slice);
        let zero = b.u_const(32, 0);
        let one = b.u_const(32, 1);
        let nonempty = b.lt(zero, len);
        let nonempty32 = b.cast_to(CastTarget::U(32), nonempty);
        let new_len = b.sub(len, nonempty32);

        emit_guarded(
            b,
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::Lt,
                lhs: *index,
                rhs: len,
            },
        );

        emit_guarded(
            b,
            guard,
            OpCode::ArrayGet {
                result: *result_elem,
                array: *slice,
                index: *index,
            },
        );

        let empty = b.mk_seq(vec![], SequenceTargetType::Slice, sel_elem_ty.clone());
        let idx_v = *index;
        let slice_v = *slice;

        let index_is_witness = index_ty.is_witness_of();
        let rebuilt = b.build_slice_extend_loop(new_len, (empty, acc_slice_ty), move |b, i| {
            let cmp_i = if idx_bits == 32 {
                i
            } else {
                b.cast_to(CastTarget::U(idx_bits), i)
            };
            let below = b.lt(cmp_i, idx_v);
            if index_is_witness {
                let next = b.add(i, one);
                let cur_val = b.array_get(slice_v, i);
                let next_val = b.array_get(slice_v, next);
                merge_select_for_slice_leaves(b, below, cur_val, next_val, &sel_elem_ty)
            } else {
                // Multiplexer
                let below32 = b.cast_to(CastTarget::U(32), below);
                let not_below = b.sub(one, below32);
                let src = b.add(i, not_below);
                b.array_get(slice_v, src)
            }
        });
        b.emit(OpCode::Cast {
            result: *result_slice,
            value: rebuilt,
            target: CastTarget::Nop,
        });
        true
    }
}
