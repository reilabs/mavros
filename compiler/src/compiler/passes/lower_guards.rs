//! Lowers `Guard` and `Select` instructions to control flow (JmpIf).
//!
//! This pass runs in the witgen pipeline, converting:
//!
//! - Guard(cond, op) → JmpIf(cond, exec_block, skip_block)
//! - Select(cond, l, r) → JmpIf(cond, t_block, f_block) → merge with phi params

use crate::compiler::{
    Field,
    analysis::types::TypeInfo,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, Instruction, Terminator, ValueId,
        hlssa::{
            CastTarget, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter, HLFunctionBuilder, HLSSABuilder},
        },
    },
};

pub struct LowerGuards {}

impl Pass for LowerGuards {
    fn name(&self) -> &'static str {
        "lower_guards"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}

impl LowerGuards {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in function_ids {
            let func_types = type_info.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                self.run_function(fb, func_types);
            });
        }
    }

    fn run_function(
        &self,
        fb: &mut HLFunctionBuilder<'_>,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        // Process blocks iteratively. We collect block IDs upfront, but new blocks
        // created during lowering don't need processing (they contain no Guards/Selects).
        let block_ids: Vec<_> = fb.function.get_blocks().map(|(bid, _)| *bid).collect();

        for block_id in block_ids {
            self.lower_block(fb, block_id, type_info);
        }
    }

    fn lower_block(
        &self,
        fb: &mut HLFunctionBuilder<'_>,
        block_id: BlockId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        // Extract instructions and terminator, then put the empty block back
        // so the block emitter can take it.
        let (instructions, terminator) = {
            let mut block = fb.function.take_block(block_id);
            let instructions = block.take_instructions();
            let terminator = block.take_terminator();
            fb.function.put_block(block_id, block);
            (instructions, terminator)
        };

        let mut emitter = fb.block(block_id);

        for instruction in instructions {
            match instruction {
                OpCode::Guard { condition, inner } => {
                    let results: Vec<_> = inner.get_results().copied().collect();
                    if results.is_empty() {
                        let (exec_block, _) = emitter.add_block();
                        let (continue_block, _) = emitter.add_block();

                        emitter.seal_and_switch(
                            Terminator::JmpIf(condition, exec_block, continue_block),
                            exec_block,
                        );
                        emitter.emit(*inner);
                        emitter.seal_and_switch(
                            Terminator::Jmp(continue_block, vec![]),
                            continue_block,
                        );
                    } else {
                        Self::lower_guard_with_result(
                            &mut emitter,
                            condition,
                            *inner,
                            &results,
                            type_info,
                        );
                    }
                }
                OpCode::Select {
                    result,
                    cond,
                    if_t,
                    if_f,
                } => {
                    let result_type = type_info.get_value_type(result).clone();

                    let (t_block, _) = emitter.add_block();
                    let (f_block, _) = emitter.add_block();
                    let (merge_block, _) = emitter.add_block();

                    // Add phi param to merge block with correct type
                    emitter
                        .function
                        .get_block_mut(merge_block)
                        .push_parameter(result, result_type);

                    // Current → JmpIf(cond, t, f)
                    emitter.seal_and_switch(Terminator::JmpIf(cond, t_block, f_block), t_block);

                    // True → Jmp(merge, if_t)
                    emitter.seal_and_switch(Terminator::Jmp(merge_block, vec![if_t]), f_block);

                    // False → Jmp(merge, if_f)
                    emitter.seal_and_switch(Terminator::Jmp(merge_block, vec![if_f]), merge_block);
                }
                other => {
                    emitter.emit(other);
                }
            }
        }

        // Apply the original terminator to the final block
        if let Some(term) = terminator {
            emitter.set_terminator(term);
        }
    }

    fn lower_guard_with_result(
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        inner: OpCode,
        results: &[ValueId],
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
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
