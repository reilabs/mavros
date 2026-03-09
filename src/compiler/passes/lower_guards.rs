use crate::compiler::{
    flow_analysis::FlowAnalysis,
    ir::r#type::Type,
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
        let block_ids: Vec<BlockId> = function.get_blocks().map(|(bid, _)| *bid).collect();

        for block_id in block_ids {
            self.lower_block(function, block_id);
        }
    }

    fn lower_block(&self, function: &mut HLFunction, block_id: BlockId) {
        let mut block = function.take_block(block_id);
        let instructions = block.take_instructions();
        let terminator = block.take_terminator();

        let mut current_block_id = block_id;
        let mut current_instructions: Vec<OpCode> = Vec::new();

        for instruction in instructions {
            match instruction {
                OpCode::Guard { condition, inner } => {
                    let results: Vec<ValueId> = inner.get_results().copied().collect();
                    let has_results = !results.is_empty();

                    // Create blocks for the conditional execution
                    let exec_block_id = function.add_block();
                    let continue_block_id = function.add_block();
                    let skip_block_id = if has_results {
                        Some(function.add_block())
                    } else {
                        None
                    };

                    let false_target = skip_block_id.unwrap_or(continue_block_id);

                    // Terminate current block with JmpIf
                    block.put_instructions(current_instructions);
                    block.set_terminator(Terminator::JmpIf(condition, exec_block_id, false_target));
                    function.put_block(current_block_id, block);

                    // Build exec block: run the inner instruction, then jump to continue
                    function
                        .get_block_mut(exec_block_id)
                        .push_instruction(*inner);
                    function
                        .get_block_mut(exec_block_id)
                        .set_terminator(Terminator::Jmp(continue_block_id, results.clone()));

                    if let Some(skip_block_id) = skip_block_id {
                        // Allocate default values upfront, then build the skip block
                        let default_values: Vec<ValueId> =
                            results.iter().map(|_| function.fresh_value()).collect();

                        for default_val in &default_values {
                            function
                                .get_block_mut(skip_block_id)
                                .push_instruction(OpCode::Const {
                                    result: *default_val,
                                    value: crate::compiler::ssa::ConstValue::Field(
                                        ark_ff::AdditiveGroup::ZERO,
                                    ),
                                });
                        }
                        function
                            .get_block_mut(skip_block_id)
                            .set_terminator(Terminator::Jmp(continue_block_id, default_values));

                        // Add phi parameters to continue block
                        for result in &results {
                            function
                                .get_block_mut(continue_block_id)
                                .push_parameter(*result, Type::field());
                        }
                    }

                    // Continue building instructions in the new continue block
                    current_block_id = continue_block_id;
                    block = function.take_block(continue_block_id);
                    current_instructions = Vec::new();
                }
                OpCode::Select {
                    result,
                    cond,
                    if_t,
                    if_f,
                } => {
                    // Select(cond, l, r) → JmpIf(cond, t_block, f_block) → merge
                    let t_block_id = function.add_block();
                    let f_block_id = function.add_block();
                    let merge_block_id = function.add_block();

                    // Terminate current block with JmpIf
                    block.put_instructions(current_instructions);
                    block.set_terminator(Terminator::JmpIf(cond, t_block_id, f_block_id));
                    function.put_block(current_block_id, block);

                    // True branch: jump to merge with if_t
                    function
                        .get_block_mut(t_block_id)
                        .set_terminator(Terminator::Jmp(merge_block_id, vec![if_t]));

                    // False branch: jump to merge with if_f
                    function
                        .get_block_mut(f_block_id)
                        .set_terminator(Terminator::Jmp(merge_block_id, vec![if_f]));

                    // Merge block: result is a phi parameter
                    function
                        .get_block_mut(merge_block_id)
                        .push_parameter(result, Type::field());

                    // Continue in merge block
                    current_block_id = merge_block_id;
                    block = function.take_block(merge_block_id);
                    current_instructions = Vec::new();
                }
                other => {
                    current_instructions.push(other);
                }
            }
        }

        // Put remaining instructions and terminator into the final block
        block.put_instructions(current_instructions);
        if let Some(term) = terminator {
            block.set_terminator(term);
        }
        function.put_block(current_block_id, block);
    }
}
