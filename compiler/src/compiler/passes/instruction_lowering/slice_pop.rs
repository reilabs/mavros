//! Lowers the remaining (pure-length) `SlicePop`s into an assert + get + copy loop.

use crate::compiler::ssa::{
    ValueId,
    hlssa::{
        CastTarget, CmpKind, OpCode, SequenceTargetType, SliceOpDir,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerSlicePop {}

impl LowerSlicePop {
    pub fn new() -> Self {
        Self {}
    }
}

impl InstructionLoweringRule for LowerSlicePop {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let (guard, op): (Option<ValueId>, &OpCode) = match instruction {
            OpCode::Guard { condition, inner } => (Some(*condition), inner.as_ref()),
            other => (None, other),
        };
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

        let len = b.slice_len(*slice);
        let zero = b.u_const(32, 0);
        let one = b.u_const(32, 1);

        let assert_nonempty = OpCode::AssertCmp {
            kind: CmpKind::Lt,
            lhs: zero,
            rhs: len,
        };
        if let Some(condition) = guard {
            b.emit(OpCode::Guard {
                condition,
                inner: Box::new(assert_nonempty),
            });
        } else {
            b.emit(assert_nonempty);
        }

        let nonempty = b.lt(zero, len);
        let nonempty32 = b.cast_to(CastTarget::U(32), nonempty);
        let new_len = b.sub(len, nonempty32);
        let elem_index = match dir {
            SliceOpDir::Back => new_len,
            SliceOpDir::Front => zero,
        };
        let get_elem = OpCode::ArrayGet {
            result: *result_elem,
            array: *slice,
            index: elem_index,
        };
        if let Some(condition) = guard {
            b.emit(OpCode::Guard {
                condition,
                inner: Box::new(get_elem),
            });
        } else {
            b.emit(get_elem);
        }

        let empty = b.mk_seq(vec![], SequenceTargetType::Slice, elem_ty);
        let slice_v = *slice;
        let front = matches!(dir, SliceOpDir::Front);
        let shrunk = b.build_slice_extend_loop(new_len, (empty, slice_ty), move |b, i| {
            let src_i = if front { b.add(i, one) } else { i };
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
