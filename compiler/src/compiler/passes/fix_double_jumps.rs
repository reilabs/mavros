//! Merges blocks connected by trivial unconditional jumps for blocks that have only one incoming
//! edge.
//!
//! This eliminates the need to jump (obviously), but also reduces block parameter traffic.

use std::collections::HashMap;

use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, Instruction, Terminator, ValueId,
        hlssa::{HLFunction, HLSSA, OpCode},
    },
};

/// Selects which value references a [`ValueReplacements`] sweep rewrites.
pub enum ReplaceScope {
    /// Only instruction *inputs* (the values an instruction reads).
    Inputs,
    /// All instruction *operands*, both inputs and results.
    Operands,
}

pub struct ValueReplacements {
    replacements: HashMap<ValueId, ValueId>,
}

impl ValueReplacements {
    pub fn new() -> Self {
        Self {
            replacements: HashMap::new(),
        }
    }

    pub fn insert(&mut self, replaced: ValueId, replacement: ValueId) {
        self.replacements.insert(replaced, replacement);
    }

    pub fn replace_instruction(&self, instruction: &mut OpCode) {
        for operand in instruction.get_operands_mut() {
            *operand = self.get_replacement(*operand);
        }
    }

    pub fn replace_inputs(&self, instruction: &mut OpCode) {
        for input in instruction.get_inputs_mut() {
            *input = self.get_replacement(*input);
        }
    }

    pub fn replace_terminator(&self, terminator: &mut Terminator) {
        match terminator {
            Terminator::Jmp(_, params) => {
                for param in params {
                    *param = self.get_replacement(*param);
                }
            }
            Terminator::JmpIf(cond, _, _) => {
                *cond = self.get_replacement(*cond);
            }
            Terminator::Return(vals) => {
                for val in vals {
                    *val = self.get_replacement(*val);
                }
            }
        }
    }

    pub fn get_replacement(&self, value: ValueId) -> ValueId {
        let mut current = value;
        for _ in 0..=self.replacements.len() {
            match self.replacements.get(&current) {
                Some(&next) if next != current => current = next,
                _ => return current,
            }
        }
        panic!("ValueReplacements: cycle starting at {:?}", value)
    }

    /// Walks every instruction and terminator in `function`, applying the collected replacements.
    ///
    /// `scope` selects whether instruction results are rewritten alongside inputs.
    pub fn apply_to_function(&self, function: &mut HLFunction, scope: ReplaceScope) {
        for (_, block) in function.get_blocks_mut() {
            for instr in block.get_instructions_mut() {
                match scope {
                    ReplaceScope::Inputs => self.replace_inputs(instr),
                    ReplaceScope::Operands => self.replace_instruction(instr),
                }
            }
            self.replace_terminator(block.get_terminator_mut());
        }
    }
}

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
            let mut replacements = HashMap::<BlockId, BlockId>::new();
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
