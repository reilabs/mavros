use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::Pass,
    passes::fix_double_jumps::ValueReplacements,
    ssa::{BlockId, SSA, Terminator, ValueId},
};

pub struct ConditionPropagation {}

impl Pass for ConditionPropagation {
    fn run(&self, ssa: &mut SSA, pass_manager: &crate::compiler::pass_manager::PassManager) {
        self.do_run(ssa, pass_manager.get_cfg());
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "condition_propagation",
            needs: vec![crate::compiler::pass_manager::DataPoint::CFG],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }
}

impl ConditionPropagation {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut SSA, cfg: &FlowAnalysis) {
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

                for (_, vid, value) in replaces {
                    let const_id = function.push_u_const(1, if *value { 1 } else { 0 });
                    replacements.insert(*vid, const_id);
                }

                let block = function.get_block_mut(block_id);
                for instruction in block.get_instructions_mut() {
                    replacements.replace_inputs(instruction);
                }
                replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }
}
