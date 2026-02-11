use std::collections::HashSet;
use crate::compiler::{
    pass_manager::{Pass, PassInfo, PassManager}, passes::fix_double_jumps::ValueReplacements, ssa::{OpCode, SSA}
};

pub struct WitnessWriteToVoid {}

impl Pass for WitnessWriteToVoid {
    fn run(&self, ssa: &mut SSA, _pass_manager: &PassManager) {
        self.do_run(ssa);
    }
    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "witness_write_to_void",
            needs: vec![],
        }
    }
    fn invalidates_cfg(&self) -> bool {
        false
    }
}

impl WitnessWriteToVoid {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut SSA) {
        let main_id = ssa.get_main_id();

        for (function_id, function) in ssa.iter_functions_mut() {
            let is_main = *function_id == main_id;
            let entry_id = function.get_entry_id();
            let mut replacements = ValueReplacements::new();

            for (block_id, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    match instruction {
                        OpCode::WriteWitness { result: r, value: b } => {
                            if let Some(r) = r {
                                replacements.insert(*r, *b);
                            }
                            *r = None;
                        }
                        _ => {}
                    }
                }
            }

            // Collect entry block param ValueIds so we can distinguish
            // entry-param WriteWitness (which must be removed) from
            // byte-decomposition WriteWitness (which must be kept).
            let entry_param_ids: HashSet<_> = if is_main {
                function.get_entry().get_parameters().map(|(id, _)| *id).collect()
            } else {
                HashSet::new()
            };

            for (block_id, block) in function.get_blocks_mut() {
                // In the main function's entry block, remove only the voided WriteWitness
                // that correspond to entry param Field→WitnessOf conversions. Their witness
                // slots overlap with input positions filled by the interpreter. Other voided
                // WriteWitness (e.g. byte decomposition from rangechecks) must be kept so
                // their values are written to the witness tape at the correct positions.
                if is_main && *block_id == entry_id {
                    let old_instructions = block.take_instructions();
                    let new_instructions = old_instructions
                        .into_iter()
                        .filter(|instr| {
                            !matches!(instr, OpCode::WriteWitness { result: None, value } if entry_param_ids.contains(value))
                        })
                        .collect();
                    block.put_instructions(new_instructions);
                }

                for instruction in block.get_instructions_mut() {
                    replacements.replace_instruction(instruction);
                }
                replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }
}
