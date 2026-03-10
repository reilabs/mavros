use crate::compiler::{
    analysis::types::{FunctionTypeInfo, TypeInfo},
    block_builder::{HLBlockEmitter, HLEmitter},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{BlockId, HLFunction, HLSSA, Instruction, OpCode, Terminator, ValueId},
};

/// Lowers Guard and Select instructions to control flow (JmpIf).
///
/// This pass runs in the witgen pipeline, converting:
/// - Guard(cond, op) → JmpIf(cond, exec_block, skip_block)
/// - Select(cond, l, r) → JmpIf(cond, t_block, f_block) → merge with phi params
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
        for function_id in function_ids {
            let mut function = ssa.take_function(function_id);
            let func_types = type_info.get_function(function_id);
            self.run_function(&mut function, func_types);
            ssa.put_function(function_id, function);
        }
    }

    fn run_function(&self, function: &mut HLFunction, type_info: &FunctionTypeInfo) {
        // Process blocks iteratively. We collect block IDs upfront, but new blocks
        // created during lowering don't need processing (they contain no Guards/Selects).
        let block_ids: Vec<_> = function.get_blocks().map(|(bid, _)| *bid).collect();

        for block_id in block_ids {
            self.lower_block(function, block_id, type_info);
        }
    }

    fn lower_block(
        &self,
        function: &mut HLFunction,
        block_id: BlockId,
        type_info: &FunctionTypeInfo,
    ) {
        // Extract instructions and terminator, then put the empty block back
        // so HLBlockEmitter can take it.
        let (instructions, terminator) = {
            let mut block = function.take_block(block_id);
            let instructions = block.take_instructions();
            let terminator = block.take_terminator();
            function.put_block(block_id, block);
            (instructions, terminator)
        };

        let mut emitter = HLBlockEmitter::new(function, block_id);

        for instruction in instructions {
            match instruction {
                OpCode::Guard { condition, inner } => {
                    let results: Vec<ValueId> = inner.get_results().copied().collect();

                    let (exec_block, _) = emitter.add_block();
                    let (continue_block, _) = emitter.add_block();

                    if results.is_empty() {
                        // No results: skip branch jumps directly to continue
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
                        let (skip_block, _) = emitter.add_block();

                        // Look up actual types for the results
                        let result_types: Vec<Type> = results
                            .iter()
                            .map(|r| type_info.get_value_type(*r).clone())
                            .collect();

                        // Add phi params to continue block with correct types
                        for (result, ty) in results.iter().zip(result_types.iter()) {
                            emitter
                                .function
                                .get_block_mut(continue_block)
                                .push_parameter(*result, ty.clone());
                        }

                        // Current → JmpIf(cond, exec, skip)
                        emitter.seal_and_switch(
                            Terminator::JmpIf(condition, exec_block, skip_block),
                            exec_block,
                        );

                        // Exec: run inner instruction, jump to continue with results
                        emitter.emit(*inner);
                        emitter.seal_and_switch(
                            Terminator::Jmp(continue_block, results),
                            skip_block,
                        );

                        // Skip: emit type-appropriate default values, jump to continue
                        let default_values: Vec<ValueId> = result_types
                            .iter()
                            .map(|ty| emit_dummy_value(ty, &mut emitter))
                            .collect();
                        emitter.seal_and_switch(
                            Terminator::Jmp(continue_block, default_values),
                            continue_block,
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
}

/// Emit a sensible default value for the given type.
fn emit_dummy_value(ty: &Type, emitter: &mut HLBlockEmitter) -> ValueId {
    match &ty.expr {
        TypeExpr::Field | TypeExpr::U(_) => emitter.field_const(ark_ff::AdditiveGroup::ZERO),
        TypeExpr::WitnessOf(inner) => {
            let dummy = emit_dummy_value(inner, emitter);
            emitter.cast_to_witness_of(dummy)
        }
        TypeExpr::Ref(inner) => {
            let dummy_inner = emit_dummy_value(inner, emitter);
            let ptr = emitter.alloc((**inner).clone());
            emitter.store(ptr, dummy_inner);
            ptr
        }
        TypeExpr::Array(elem, len) => {
            let dummy_elem = emit_dummy_value(elem, emitter);
            let elems = vec![dummy_elem; *len];
            emitter.mk_seq(elems, crate::compiler::ssa::SeqType::Array(*len), (**elem).clone())
        }
        TypeExpr::Slice(elem) => {
            // Empty-ish slice: single dummy element
            let dummy_elem = emit_dummy_value(elem, emitter);
            emitter.mk_seq(
                vec![dummy_elem],
                crate::compiler::ssa::SeqType::Slice,
                (**elem).clone(),
            )
        }
        TypeExpr::Tuple(fields) => {
            let dummy_elems: Vec<ValueId> = fields
                .iter()
                .map(|f| emit_dummy_value(f, emitter))
                .collect();
            emitter.mk_tuple(dummy_elems, fields.clone())
        }
        TypeExpr::Function => emitter.field_const(ark_ff::AdditiveGroup::ZERO),
    }
}
