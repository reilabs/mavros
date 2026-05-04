use crate::compiler::{
    analysis::types::TypeInfo,
    block_builder::{HLBlockEmitter, HLEmitter},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{BlockId, HLFunction, HLSSA, Instruction, OpCode, Terminator},
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

    fn run_function(
        &self,
        function: &mut HLFunction,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        // Process blocks iteratively. We collect block IDs upfront, but new blocks
        // created during lowering don't need processing (they contain no
        // Guards/Selects).
        let block_ids: Vec<_> = function.get_blocks().map(|(bid, _)| *bid).collect();

        for block_id in block_ids {
            self.lower_block(function, block_id, type_info);
        }
    }

    fn lower_block(
        &self,
        function: &mut HLFunction,
        block_id: BlockId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
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
                    let results: Vec<_> = inner.get_results().copied().collect();
                    assert!(
                        results.is_empty(),
                        "ICE: Guard with results is not supported in LowerGuards"
                    );

                    let (exec_block, _) = emitter.add_block();
                    let (continue_block, _) = emitter.add_block();

                    emitter.seal_and_switch(
                        Terminator::JmpIf(condition, exec_block, continue_block),
                        exec_block,
                    );
                    emitter.emit(*inner);
                    emitter
                        .seal_and_switch(Terminator::Jmp(continue_block, vec![]), continue_block);
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
