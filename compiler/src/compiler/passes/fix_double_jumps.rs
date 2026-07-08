//! Merges blocks connected by trivial unconditional jumps for blocks that have only one incoming
//! edge.
//!
//! This eliminates the need to jump (obviously), but also reduces block parameter traffic.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::flow_analysis::FlowAnalysis,
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        passes::shared::value_replacements::{ReplaceScope, ValueReplacements},
        ssa::{BlockId, Terminator, hlssa::HLSSA},
    },
};

pub struct FixDoubleJumps {}

impl Pass for FixDoubleJumps {
    fn name(&self) -> &'static str {
        "fix_double_jumps"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<FlowAnalysis>());
    }
}

impl FixDoubleJumps {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, flow_analysis: &FlowAnalysis) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let cfg = flow_analysis.get_function_cfg(*function_id);
            let jumps = cfg.find_redundant_jumps();
            let mut replacements = HashMap::<BlockId, BlockId>::default();
            let mut value_replacements = ValueReplacements::new();
            for (mut source, mut target) in jumps {
                while let Some(src) = replacements.get(&source) {
                    source = *src
                }
                while let Some(tgt) = replacements.get(&target) {
                    target = *tgt;
                }
                let mut target_block = function.take_block(target);
                let source_block = function.get_block_mut(source);

                let jump_args = match source_block.get_terminator() {
                    Some(Terminator::Jmp(_, params)) => params.clone(),
                    _ => panic!("ICE: CFG says there is a jump here"),
                };

                for ((param, _), arg) in target_block.get_parameters().zip(jump_args) {
                    value_replacements.insert(*param, arg);
                }

                for instruction in target_block.take_instructions() {
                    source_block.push_instruction(instruction);
                }

                source_block.set_terminator(target_block.take_terminator().unwrap());
                replacements.insert(target, source);
            }

            value_replacements.apply_to_function(function, ReplaceScope::Operands);
        }
    }
}
