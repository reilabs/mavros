//! Lowers a witness `Select` on a slice

use crate::compiler::ssa::hlssa::{
    CastTarget, OpCode, SequenceTargetType,
    builder::{HLBlockEmitter, HLEmitter},
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerSliceSelect {}

impl LowerSliceSelect {
    pub fn new() -> Self {
        Self {}
    }
}

impl InstructionLoweringRule for LowerSliceSelect {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let OpCode::Select {
            result,
            cond,
            if_t,
            if_f,
        } = instruction
        else {
            return false;
        };
        let result_ty = context.types().get_value_type(*result).clone();
        if !result_ty.is_slice() {
            return false;
        }
        let elem_ty = result_ty.get_array_element();
        let cond = *cond;
        let a = *if_t;
        let c = *if_f;

        let len_a = b.slice_len(a);
        let len_c = b.slice_len(c);
        let a_shorter = b.lt(len_a, len_c);
        let min = b.select(a_shorter, len_a, len_c);
        let acc = b.mk_seq(vec![], SequenceTargetType::Slice, elem_ty.clone());

        // Prefix `0 .. min(len_a, len_c)`
        let acc = b.build_slice_extend_loop(min, (acc, result_ty.clone()), |b, i| {
            let a_i = b.array_get(a, i);
            let c_i = b.array_get(c, i);
            b.select(cond, a_i, c_i)
        });

        // Suffix from `a` (`len(acc) .. len_a`): runs only when `a` is the longer arm.
        let acc =
            b.build_slice_extend_loop(len_a, (acc, result_ty.clone()), |b, i| b.array_get(a, i));

        // Suffix from `c` (`len(acc) .. len_c`): runs only when `c` is the longer arm
        let acc =
            b.build_slice_extend_loop(len_c, (acc, result_ty.clone()), |b, i| b.array_get(c, i));

        b.emit(OpCode::Cast {
            result: *result,
            value: acc,
            target: CastTarget::Nop,
        });
        true
    }
}
