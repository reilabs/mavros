use std::collections::HashMap;

use crate::compiler::{
    flow_analysis::FlowAnalysis,
    pass_manager::{DataPoint, Pass, PassInfo, PassManager},
    ssa::{BlockId, OpCode, SSA, Terminator, ValueId},
};

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
        let replacements_replacement = self.replacements.get(&replacement).unwrap_or(&replacement);
        self.replacements
            .insert(replaced, *replacements_replacement);
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
        let replacement = self.replacements.get(&value).unwrap_or(&value);
        *replacement
    }
}

pub struct FixDoubleJumps {}

impl Pass for FixDoubleJumps {
    fn run(&self, ssa: &mut SSA, pass_manager: &PassManager) {
        let flow_analysis = pass_manager.get_cfg();
        self.do_run(ssa, flow_analysis);
    }

    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "fix_double_jumps",
            needs: vec![DataPoint::CFG],
        }
    }
}

impl FixDoubleJumps {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut SSA, flow_analysis: &FlowAnalysis) {
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

            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    value_replacements.replace_instruction(instruction);
                }
                let mut terminator = block.take_terminator().unwrap();
                value_replacements.replace_terminator(&mut terminator);
                block.set_terminator(terminator);
            }
        }
    }
}
