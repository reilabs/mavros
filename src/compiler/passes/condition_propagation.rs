use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{BlockId, HLSSA, OpCode, Terminator, ValueId},
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
        for (function_id, function) in ssa.iter_functions_mut() {
            let mut replaces: Vec<(BlockId, ValueId, bool)> = vec![];

            for (_, block) in function.get_blocks() {
                match block.get_terminator() {
                    Some(Terminator::JmpIf(cond, then_block, else_block)) => {
                        replaces.push((*then_block, *cond, true));
                        replaces.push((*else_block, *cond, false));
                    }
                    _ => {}
                }
            }

            let cfg = cfg.get_function_cfg(*function_id);

            for block_id in cfg.get_domination_pre_order() {
                let mut replacements = ValueReplacements::new();
                let replaces = replaces
                    .iter()
                    .filter(|(cond_block, _, _)| cfg.dominates(*cond_block, block_id));

                let mut const_opcodes = Vec::new();
                for (_, vid, value) in replaces {
                    let const_id = function.fresh_value();
                    const_opcodes.push(OpCode::mk_u_const(const_id, 1, if *value { 1 } else { 0 }));
                    replacements.insert(*vid, const_id);
                }

                let block = function.get_block_mut(block_id);
                let mut instructions = block.take_instructions();
                let mut new_instructions = const_opcodes;
                new_instructions.extend(instructions.iter().cloned());
                for instruction in new_instructions.iter_mut() {
                    replacements.replace_inputs(instruction);
                }
                block.put_instructions(new_instructions);
                replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }
}
