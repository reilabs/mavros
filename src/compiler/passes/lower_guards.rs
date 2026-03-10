use crate::compiler::{
    block_builder::{HLBlockEmitter, HLEmitter},
    ir::r#type::Type,
    pass_manager::{AnalysisId, AnalysisStore, Pass},
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
    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}

impl LowerGuards {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut HLSSA) {
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            let mut function = ssa.take_function(function_id);
            self.run_function(&mut function);
            ssa.put_function(function_id, function);
        }
    }

    fn run_function(&self, function: &mut HLFunction) {
        // Process blocks iteratively. We collect block IDs upfront, but new blocks
        // created during lowering don't need processing (they contain no Guards/Selects).
        let block_ids: Vec<_> = function.get_blocks().map(|(bid, _)| *bid).collect();

        for block_id in block_ids {
            self.lower_block(function, block_id);
        }
    }

    fn lower_block(&self, function: &mut HLFunction, block_id: BlockId) {
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

                        // Add phi params to continue block
                        for result in &results {
                            emitter
                                .function
                                .get_block_mut(continue_block)
                                .push_parameter(*result, Type::field());
                        }

                        let num_results = results.len();

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

                        // Skip: emit default Field(0) values, jump to continue
                        let default_values: Vec<ValueId> = (0..num_results)
                            .map(|_| {
                                emitter.field_const(ark_ff::AdditiveGroup::ZERO)
                            })
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
                    let (t_block, _) = emitter.add_block();
                    let (f_block, _) = emitter.add_block();
                    let (merge_block, _) = emitter.add_block();

                    // Add phi param to merge block
                    emitter
                        .function
                        .get_block_mut(merge_block)
                        .push_parameter(result, Type::field());

                    // Current → JmpIf(cond, t, f)
                    emitter.seal_and_switch(
                        Terminator::JmpIf(cond, t_block, f_block),
                        t_block,
                    );

                    // True → Jmp(merge, if_t)
                    emitter.seal_and_switch(
                        Terminator::Jmp(merge_block, vec![if_t]),
                        f_block,
                    );

                    // False → Jmp(merge, if_f)
                    emitter.seal_and_switch(
                        Terminator::Jmp(merge_block, vec![if_f]),
                        merge_block,
                    );
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
