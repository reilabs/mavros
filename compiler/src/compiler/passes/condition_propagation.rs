//! Performs constant propagation of known condition values into downstream blocks.

use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{
        BlockId, Terminator, ValueId,
        hlssa::{ConstValue, HLSSA, OpCode, builder::HLSSABuilder},
    },
};

pub struct ConditionPropagation {}

impl Pass for ConditionPropagation {
    fn name(&self) -> &'static str {
        "condition_propagation"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<FlowAnalysis>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl ConditionPropagation {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, cfg: &FlowAnalysis) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let func_cfg = cfg.get_function_cfg(function_id);
            sb.modify_function(function_id, |fb| {
                let mut replaces: Vec<(BlockId, ValueId, bool)> = vec![];

                for (_, block) in fb.function.get_blocks() {
                    if let Some(Terminator::JmpIf(cond, then_b, else_b)) = block.get_terminator() {
                        replaces.push((*then_b, *cond, true));
                        replaces.push((*else_b, *cond, false));
                    }
                }

                for block_id in func_cfg.get_domination_pre_order() {
                    let mut replacements = ValueReplacements::new();
                    let dominated = replaces
                        .iter()
                        .filter(|(cond_block, _, _)| func_cfg.dominates(*cond_block, block_id));

                    let mut const_opcodes = Vec::new();
                    for (_, vid, value) in dominated {
                        let const_id = fb.ssa.fresh_value();
                        const_opcodes.push(OpCode::Const {
                            result: const_id,
                            value: ConstValue::U(1, if *value { 1 } else { 0 }),
                        });
                        replacements.insert(*vid, const_id);
                    }

                    let block = fb.function.get_block_mut(block_id);
                    let instructions = block.take_instructions();
                    let mut new_instructions = const_opcodes;
                    new_instructions.extend(instructions.iter().cloned());
                    for instruction in new_instructions.iter_mut() {
                        replacements.replace_inputs(instruction);
                    }
                    block.put_instructions(new_instructions);
                    replacements.replace_terminator(block.get_terminator_mut());
                }
            });
        }
    }
}
