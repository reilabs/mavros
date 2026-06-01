//! Lowers `Guard` and `Select` instructions to control flow (JmpIf).
//!
//! This pass runs in the witgen pipeline, converting:
//!
//! - Guard(cond, op) → JmpIf(cond, exec_block, skip_block)
//! - Select(cond, l, r) → JmpIf(cond, t_block, f_block) → merge with phi params

use crate::compiler::{
    Field,
    analysis::types::FunctionTypeInfo,
    ssa::{
        Instruction, ValueId,
        hlssa::{
            CastTarget, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::lowering_pass::{LoweringContext, LoweringPass};

pub struct LowerGuards {}

impl LoweringPass for LowerGuards {
    const NAME: &'static str = "lower_guards";

    fn process_instruction(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: OpCode,
    ) {
        let type_info = context.types();
        match instruction {
            OpCode::Guard { condition, inner } => {
                let results: Vec<_> = inner.get_results().copied().collect();
                if results.is_empty() {
                    emitter.build_if_else_into(
                        condition,
                        vec![],
                        |e| {
                            e.emit(*inner);
                            vec![]
                        },
                        |_| vec![],
                    );
                } else {
                    Self::lower_guard_with_result(emitter, condition, *inner, &results, type_info);
                }
            }
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let result_type = type_info.get_value_type(result).clone();
                emitter.build_if_else_into(
                    cond,
                    vec![(result, result_type)],
                    |_| vec![if_t],
                    |_| vec![if_f],
                );
            }
            other => {
                emitter.emit(other);
            }
        }
    }
}

impl LowerGuards {
    pub fn new() -> Self {
        Self {}
    }

    fn lower_guard_with_result(
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        inner: OpCode,
        results: &[ValueId],
        type_info: &FunctionTypeInfo,
    ) {
        let result_pairs: Vec<(ValueId, Type)> = results
            .iter()
            .map(|r| (*r, type_info.get_value_type(*r).clone()))
            .collect();
        emitter.build_if_else_into(
            condition,
            result_pairs.clone(),
            |e| {
                let mut inner_fresh = inner.clone();
                let fresh_results: Vec<_> = inner_fresh
                    .get_results_mut()
                    .map(|result| {
                        let fresh = e.fresh_value();
                        *result = fresh;
                        fresh
                    })
                    .collect();
                e.emit(inner_fresh);
                fresh_results
            },
            |e| {
                result_pairs
                    .iter()
                    .map(|(_, ty)| Self::default_value(e, ty))
                    .collect()
            },
        );
    }

    fn default_value(emitter: &mut HLBlockEmitter<'_>, ty: &Type) -> ValueId {
        match &ty.expr {
            TypeExpr::Field => emitter.field_const(Field::from(0u64)),
            TypeExpr::U(bits) => emitter.u_const(*bits, 0),
            TypeExpr::I(bits) => emitter.i_const(*bits, 0),
            TypeExpr::WitnessOf(inner) => {
                let inner_val = Self::default_value(emitter, inner);
                emitter.cast_to(CastTarget::WitnessOf, inner_val)
            }
            TypeExpr::Array(elem, n) => {
                let elem_default = Self::default_value(emitter, elem);
                emitter.mk_repeated(
                    elem_default,
                    SequenceTargetType::Array(*n),
                    *n,
                    (**elem).clone(),
                )
            }
            other => panic!(
                "LowerGuards: cannot synthesize default value for type {:?}",
                other
            ),
        }
    }
}
