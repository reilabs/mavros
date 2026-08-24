//! Lowers the remaining (pure-length) `SlicePop`s into an assert + get + copy loop.

use crate::compiler::{
    passes::{
        instruction_lowering::{InstructionLoweringRule, LoweringContext},
        shared::seq_bounds::build_pop_bounds_assert,
    },
    ssa::hlssa::{
        CastTarget, OpCode, SequenceTargetType, SliceOpDir,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

#[derive(Default)]
pub struct LowerSlicePop;

impl InstructionLoweringRule for LowerSlicePop {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (guard, op) = HLBlockEmitter::unwrap_guard(instruction);
        let OpCode::SlicePop {
            dir,
            result_slice,
            result_elem,
            slice,
        } = op
        else {
            return false;
        };

        let slice_ty = context.types().get_value_type(*slice).clone();
        let elem_ty = slice_ty.get_array_element();

        let (assert, len) = build_pop_bounds_assert(b, *slice);
        b.emit_guarded(guard, assert);

        let zero = b.int_const(32, 0);
        let one = b.int_const(32, 1);
        let nonempty = b.ult(zero, len);
        let nonempty32 = b.cast_to(CastTarget::Int(32), nonempty);
        let new_len = b.usub(len, nonempty32);
        let elem_index = match dir {
            SliceOpDir::Back => new_len,
            SliceOpDir::Front => zero,
        };
        b.emit_guarded(
            guard,
            OpCode::ArrayGet {
                result: *result_elem,
                array: *slice,
                index: elem_index,
            },
        );

        let empty = b.mk_seq(vec![], SequenceTargetType::Slice, elem_ty);
        let slice_v = *slice;
        let front = matches!(dir, SliceOpDir::Front);
        let shrunk = b.build_slice_extend_loop(new_len, (empty, slice_ty), move |b, i| {
            let src_i = if front { b.uadd(i, one) } else { i };
            b.array_get(slice_v, src_i)
        });
        b.emit(OpCode::Cast {
            result: *result_slice,
            value: shrunk,
            target: CastTarget::Nop,
        });
        true
    }
}
