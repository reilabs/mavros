//! Lowers a witness `Select` on a slice

use crate::compiler::ssa::hlssa::{
    CastTarget, OpCode, SequenceTargetType, SliceOpDir, Type,
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
        let zero = b.u_const(32, 0);
        let one = b.u_const(32, 1);
        let acc = b.mk_seq(vec![], SequenceTargetType::Slice, elem_ty.clone());

        // Prefix `0 .. min(len_a, len_c)`
        let acc = b.build_loop(
            vec![(zero, Type::u(32)), (acc, result_ty.clone())],
            |hb, p| {
                let in_a = hb.lt(p[0], len_a);
                let in_c = hb.lt(p[0], len_c);
                hb.and(in_a, in_c)
            },
            |bb, p| {
                let i = p[0];
                let a_i = bb.array_get(a, i);
                let c_i = bb.array_get(c, i);
                let sel = bb.select(cond, a_i, c_i);
                let acc = bb.slice_push(p[1], vec![sel], SliceOpDir::Back);
                vec![bb.add(i, one), acc]
            },
        )[1];

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
